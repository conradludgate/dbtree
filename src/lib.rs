use std::{io, marker::PhantomData};

use array::Array;
use checksum::{validate_page, validate_page_mut};
use source::{PageSink, PageSource};
use zerocopy::{little_endian, FromBytes, FromZeros, Immutable, IntoBytes, KnownLayout, Unaligned};

mod checksum;
mod dbg;
pub mod source;

/// A BTree
///
/// * `S` (a [`PageSource`]/[`PageSink`]) defines where the data is stored
/// * `F` (a [`Factor`]) defines the page structure of the tree
pub struct BTree<S, F: Factor> {
    page_store: S,
    factor: PhantomData<F>,
    metadata: FileMetadata,
}

mod array {
    use std::ops::{Index, IndexMut, Range, RangeFrom, RangeTo};

    use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

    pub trait Array<T>:
        IntoBytes
        + FromBytes
        + KnownLayout
        + Unaligned
        + Immutable
        + AsRef<[T]>
        + AsMut<[T]>
        + Index<RangeFrom<usize>, Output = [T]>
        + Index<RangeTo<usize>, Output = [T]>
        + Index<Range<usize>, Output = [T]>
        + Index<usize, Output = T>
        + IndexMut<RangeFrom<usize>, Output = [T]>
        + IndexMut<RangeTo<usize>, Output = [T]>
        + IndexMut<Range<usize>, Output = [T]>
        + IndexMut<usize, Output = T>
    {
        const SIZE: usize;
    }

    impl<T: IntoBytes + FromBytes + KnownLayout + Unaligned + Immutable, const N: usize> Array<T>
        for [T; N]
    {
        const SIZE: usize = N;
    }
}

pub trait Factor {
    type Page: Array<u8>;
    type Key: Copy + Ord + IntoBytes + FromBytes + KnownLayout + Unaligned + Immutable;

    type Branch<T: IntoBytes + FromBytes + KnownLayout + Unaligned + Immutable>: Array<T>;
}

pub struct DefaultTreeFactors;

impl Factor for DefaultTreeFactors {
    type Page = [u8; 65536];
    type Key = [u8; 128];

    type Branch<T: IntoBytes + FromBytes + KnownLayout + Unaligned + Immutable> = [T; 480];
}

fn onecopy<T: IntoBytes + FromBytes + Immutable>(t: &T) -> T {
    T::read_from_bytes(t.as_bytes()).unwrap()
}

impl<S: PageSource<F::Page>, F: Factor> BTree<S, F> {
    pub fn new(s: S) -> io::Result<Self> {
        let metadata;
        if let Some(page) = s.read(NODE_SENTINAL)? {
            metadata = *validate_page(&page)?;
        } else {
            // zero value represents an empty tree.
            metadata = FileMetadata::new_zeroed();
        }

        Ok(Self {
            page_store: s,
            factor: PhantomData,
            metadata,
        })
    }

    fn search_internal_node(&self, node_ref: HeapPtr, key: &F::Key) -> io::Result<HeapPtr> {
        let page = self.page_store.must_read(node_ref)?;
        let node = validate_page::<InternalNode<F>, F::Page>(&page)?;

        let len = node.len.get() as usize;
        let keys = &node.keys[..len];
        match keys.binary_search(key) {
            Err(0) => Ok(node.min),
            Ok(i) => Ok(node.values[i]),
            Err(i) => Ok(node.values[i - 1]),
        }
    }

    fn search_leaf_node(&self, node: HeapPtr, key: &F::Key) -> io::Result<SearchEntry> {
        let page = self.page_store.must_read(node)?;
        let node = validate_page::<LeafNode<F>, F::Page>(&page)?;

        let len = node.len.get() as usize;
        let keys = &node.keys[..len];
        match keys.binary_search(key) {
            Ok(i) => Ok(SearchEntry::Occupied(i, node.values[i])),
            Err(i) => Ok(SearchEntry::Vacant(i)),
        }
    }

    pub fn search(&self, key: &F::Key) -> io::Result<Option<HeapPtr>> {
        let mut depth = self.metadata.depth.get();
        let mut current = self.metadata.root;

        // the btree is currently empty.
        if current == NODE_SENTINAL {
            return Ok(None);
        }

        // walk through the internal nodes until we find the correct leaf
        while depth > 0 {
            current = self.search_internal_node(current, key)?;
            depth -= 1;
        }
        match self.search_leaf_node(current, key)? {
            SearchEntry::Occupied(_, node_ref) => Ok(Some(node_ref)),
            SearchEntry::Vacant(_) => Ok(None),
        }
    }
}

enum InsertState<F: Factor> {
    Inserted,
    Replaced(HeapPtr),
    Split(F::Key, HeapPtr),
}

enum InsertInternalState<F: Factor> {
    Inserted,
    Split(F::Key, HeapPtr),
}

impl<S, F> BTree<S, F>
where
    S: PageSink<F::Page>,
    F: Factor,
{
    fn allocate(&mut self) -> io::Result<HeapPtr> {
        // todo: use freelist

        // skip the first page reserved for metadata.
        if self.metadata.len == 0 {
            self.metadata.len += 1;
        }

        let index = self.metadata.len.get();
        let page_size = u64::try_from(F::Page::SIZE).unwrap();
        let offset = index
            .checked_mul(page_size)
            .ok_or_else(|| io::Error::new(io::ErrorKind::OutOfMemory, "page offset overflowed u64"))?;

        self.metadata.len += 1;
        self.update_metadata()?;

        Ok(HeapPtr {
            offset: little_endian::U64::new(offset),
        })
    }

    fn update_metadata(&mut self) -> io::Result<()> {
        let mut page = F::Page::new_zeroed();
        self.metadata.write_to_prefix(page.as_mut_bytes()).unwrap();
        self.page_store.write_page(NODE_SENTINAL, &mut page)
    }

    fn insert_leaf_node(
        &mut self,
        page_ref: HeapPtr,
        key: &F::Key,
        value: HeapPtr,
    ) -> io::Result<InsertState<F>> {
        let mut page = self.page_store.must_read(page_ref)?;
        let node = validate_page_mut::<LeafNode<F>, F::Page>(&mut page)?;

        let len = node.len.get() as usize;
        let keys = &node.keys[..len];

        match keys.binary_search(key) {
            Ok(i) => {
                let old_value = std::mem::replace(&mut node.values[i], value);
                self.page_store.write_page(page_ref, &mut page)?;
                Ok(InsertState::Replaced(old_value))
            }
            Err(i) if len < F::Branch::<u8>::SIZE => {
                node.insert(i, *key, value);
                self.page_store.write_page(page_ref, &mut page)?;
                Ok(InsertState::Inserted)
            }
            // overflow
            Err(_) => {
                let new_page_ref = self.allocate()?;
                let mut new_page = F::Page::new_zeroed();

                let pivot = {
                    let (new_node, _) =
                        LeafNode::<F>::mut_from_prefix(new_page.as_mut_bytes()).unwrap();

                    node.split_into(new_node)
                };

                self.page_store.write_page(page_ref, &mut page)?;
                self.page_store.write_page(new_page_ref, &mut new_page)?;

                Ok(InsertState::Split(pivot, new_page_ref))
            }
        }
    }

    fn must_insert_leaf_node(
        &mut self,
        page_ref: HeapPtr,
        key: &F::Key,
        value: HeapPtr,
    ) -> io::Result<()> {
        let mut page = self.page_store.must_read(page_ref)?;
        let node = validate_page_mut::<LeafNode<F>, F::Page>(&mut page)?;

        let len = node.len.get() as usize;
        let keys = &node.keys[..len];

        match keys.binary_search(key) {
            Ok(_) => unreachable!(),
            Err(i) if len < F::Branch::<u8>::SIZE => {
                node.insert(i, *key, value);
                self.page_store.write_page(page_ref, &mut page)?;
                Ok(())
            }
            Err(_) => unreachable!("btree insert must not overflow"),
        }
    }

    fn insert_internal_node(
        &mut self,
        page_ref: HeapPtr,
        key: &F::Key,
        value: HeapPtr,
    ) -> io::Result<InsertInternalState<F>> {
        let mut page = self.page_store.must_read(page_ref)?;
        let node = validate_page_mut::<InternalNode<F>, F::Page>(&mut page)?;

        let len = node.len.get() as usize;
        let keys = &node.keys[..len];

        match keys.binary_search(key) {
            Ok(_) => unreachable!("if we are inserting an internal node, it is because this key wasn't in the tree and a leaf node had to split."),
            Err(i) if len < F::Branch::<u8>::SIZE => {
                node.insert(i, *key, value);
                self.page_store.write_page(page_ref, &mut page)?;
                Ok(InsertInternalState::Inserted)
            }
            // overflow
            Err(_) => {
                let new_page_ref = self.allocate()?;
                let mut new_page = F::Page::new_zeroed();

                let pivot = {
                    let (new_node, _) =
                    InternalNode::<F>::mut_from_prefix(new_page.as_mut_bytes()).unwrap();

                    node.split_into(new_node)
                };

                self.page_store.write_page(page_ref, &mut page)?;
                self.page_store.write_page(new_page_ref, &mut page)?;

                Ok(InsertInternalState::Split(pivot, new_page_ref))
            }
        }
    }

    fn must_insert_internal_node(
        &mut self,
        page_ref: HeapPtr,
        key: &F::Key,
        value: HeapPtr,
    ) -> io::Result<()> {
        let mut page = self.page_store.must_read(page_ref)?;
        let node = validate_page_mut::<InternalNode<F>, F::Page>(&mut page)?;

        let len = node.len.get() as usize;
        let keys = &node.keys[..len];

        match keys.binary_search(key) {
            Ok(_) => unreachable!("key should not already be here"),
            Err(i) if len < F::Branch::<u8>::SIZE => {
                node.insert(i, *key, value);
                self.page_store.write_page(page_ref, &mut page)?;
                Ok(())
            }
            Err(_) => unreachable!("btree insert must not overflow"),
        }
    }

    // returns the old NodeRef if there was one.
    pub fn insert(&mut self, key: &F::Key, value: HeapPtr) -> io::Result<Option<HeapPtr>> {
        let mut depth = self.metadata.depth.get();
        let mut current = self.metadata.root;

        // the btree is currently empty.
        // allocate a new empty root.
        if current == NODE_SENTINAL {
            let root_page_ref = self.allocate()?;

            let mut page = F::Page::new_zeroed();
            {
                let (_root_node, _) = LeafNode::<F>::mut_from_prefix(page.as_mut_bytes()).unwrap();
                // zero value for the leaf is valid as empty
            }

            self.page_store.write_page(root_page_ref, &mut page)?;

            self.metadata.root = root_page_ref;
            self.update_metadata()?;

            current = root_page_ref;
        }

        // walk through the internal nodes until we find the correct leaf
        let mut stack = vec![];
        while depth > 0 {
            stack.push(current);
            current = self.search_internal_node(current, key)?;
            depth -= 1;
        }

        // try insert into the leaf
        let (mut pivot, mut new_node_ref) = match self.insert_leaf_node(current, key, value)? {
            InsertState::Inserted => return Ok(None),
            InsertState::Replaced(node_ref) => return Ok(Some(node_ref)),
            InsertState::Split(pivot, new_node_ref) => (pivot, new_node_ref),
        };

        // the node had to split, try insert into the parent.
        let mut stack2 = vec![(*key, value, depth)];
        current = loop {
            depth += 1;

            let Some(parent) = stack.pop() else {
                // no more parents, we need to allocate a new root.
                let root_page_ref = self.allocate()?;
                {
                    let mut page = F::Page::new_zeroed();
                    {
                        let (root_node, _) =
                            InternalNode::<F>::mut_from_prefix(page.as_mut_bytes()).unwrap();

                        root_node.len.set(1);
                        root_node.min = self.metadata.root;
                        root_node.keys[0] = pivot;
                        root_node.values[0] = new_node_ref;
                    }

                    self.page_store.write_page(root_page_ref, &mut page)?;
                }

                self.metadata.root = root_page_ref;
                self.metadata.depth += 1;
                self.update_metadata()?;

                break root_page_ref;
            };

            // insert into the parent, this might split again
            match self.insert_internal_node(parent, &pivot, new_node_ref)? {
                InsertInternalState::Inserted => break parent,
                InsertInternalState::Split(p, n) => {
                    stack2.push((pivot, new_node_ref, depth));
                    (pivot, new_node_ref) = (p, n)
                }
            };
        };

        // walk back down the new set of internal nodes
        drop(stack);
        loop {
            let Some((key, value, d)) = stack2.pop() else {
                break Ok(None);
            };

            while depth > d {
                current = self.search_internal_node(current, &key)?;
                depth -= 1;
            }

            if d > 0 {
                // insert into the leaf that now is guaranteed to have space
                self.must_insert_internal_node(current, &key, value)?;
            } else {
                // insert into the leaf that now is guaranteed to have space
                self.must_insert_leaf_node(current, &key, value)?;
            }
        }
    }
}

pub enum SearchEntry {
    Occupied(usize, HeapPtr),
    Vacant(usize),
}

#[derive(KnownLayout, IntoBytes, Unaligned, FromBytes, Immutable)]
#[repr(C)]
pub struct InternalNode<F: Factor> {
    keys: F::Branch<F::Key>,
    min: HeapPtr,
    values: F::Branch<HeapPtr>,

    len: little_endian::U16,
}

impl<F: Factor> InternalNode<F> {
    fn split_into(&mut self, rhs: &mut Self) -> F::Key {
        debug_assert_eq!(rhs.len, 0);
        debug_assert_eq!(self.len.get() as usize, F::Branch::<u8>::SIZE);

        let len_l = F::Branch::<u8>::SIZE / 2;
        let len_r = F::Branch::<u8>::SIZE - len_l;

        self.len.set(len_l as u16);
        rhs.len.set(len_r as u16);

        let (_, keys_r) = self.keys.as_ref().split_at(len_l);
        let (keys_p, keys_r) = keys_r.split_first().unwrap();

        let (_, vals_r) = self.values.as_ref().split_at(len_l);
        let (vals_p, vals_r) = vals_r.split_first().unwrap();

        rhs.min = *vals_p;
        rhs.keys[..len_r].copy_from_slice(keys_r);
        rhs.values[..len_r].copy_from_slice(vals_r);

        *keys_p
    }

    fn insert(&mut self, i: usize, key: F::Key, value: HeapPtr) {
        self.keys[i..].rotate_right(1);
        self.values[i..].rotate_right(1);

        self.keys[i] = key;
        self.values[i] = value;

        self.len += 1;
    }
}

#[derive(KnownLayout, IntoBytes, Unaligned, FromBytes, Immutable)]
#[repr(C)]
pub struct LeafNode<F: Factor> {
    keys: F::Branch<F::Key>,
    values: F::Branch<HeapPtr>,

    // /// represents a doubly linked list of leaf nodes.
    // /// we only had space for 1 pointer, so this is an xor link.
    // xor_link: NodeRef,
    len: little_endian::U16,
}

impl<F: Factor> LeafNode<F> {
    fn split_into(&mut self, rhs: &mut Self) -> F::Key {
        debug_assert_eq!(rhs.len, 0);
        debug_assert_eq!(self.len.get() as usize, F::Branch::<u8>::SIZE);

        let len_l = F::Branch::<u8>::SIZE / 2;
        let len_r = F::Branch::<u8>::SIZE - len_l;

        self.len.set(len_l as u16);
        rhs.len.set(len_r as u16);

        let (_, keys_r) = self.keys.as_ref().split_at(len_l);
        let (keys_p, _) = keys_r.split_first().unwrap();

        let (_, vals_r) = self.values.as_ref().split_at(len_l);

        rhs.keys[..len_r].copy_from_slice(keys_r);
        rhs.values[..len_r].copy_from_slice(vals_r);

        *keys_p
    }

    fn insert(&mut self, i: usize, key: F::Key, value: HeapPtr) {
        self.keys[i..].rotate_right(1);
        self.values[i..].rotate_right(1);

        self.keys[i] = key;
        self.values[i] = value;
        self.len += 1;
    }

    // fn next_leaf(&self, prev_ref: NodeRef) -> NodeRef {
    //     NodeRef {
    //         offset: little_endian::U64::new(self.xor_link.offset.get() ^ prev_ref.offset.get()),
    //     }
    // }

    // fn prev_leaf(&self, next_ref: NodeRef) -> NodeRef {
    //     NodeRef {
    //         offset: little_endian::U64::new(self.xor_link.offset.get() ^ next_ref.offset.get()),
    //     }
    // }
}

#[derive(KnownLayout, IntoBytes, Unaligned, FromBytes, Immutable, Clone, Copy)]
#[repr(C)]
struct FileMetadata {
    // how many pages are allocated in this file
    len: little_endian::U64,

    // the tree root
    root: HeapPtr,
    depth: little_endian::U64,

    // a doubly linked list for the free pages
    free_head: HeapPtr,
    free_tail: HeapPtr,
}

/// Reserved for the file metadata.
/// In any other position it is a "None" value.
/// Can be used to indicate that no entry is present.
const NODE_SENTINAL: HeapPtr = HeapPtr {
    offset: little_endian::U64::new(0),
};

#[derive(KnownLayout, IntoBytes, Unaligned, Immutable, FromBytes, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct HeapPtr {
    offset: little_endian::U64,
}

#[derive(KnownLayout, IntoBytes, Unaligned, Immutable, FromBytes, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct NodePtr(HeapPtr);

#[derive(KnownLayout, IntoBytes, Unaligned, Immutable, FromBytes, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct LeafPtr(NodePtr);

#[derive(KnownLayout, IntoBytes, Unaligned, Immutable, FromBytes, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct InternalPtr(NodePtr);

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rand::{rngs::StdRng, Rng, SeedableRng};
    use zerocopy::little_endian;

    use crate::{BTree, DefaultTreeFactors, Factor, HeapPtr};

    pub fn node(n: u64) -> HeapPtr {
        crate::HeapPtr {
            offset: little_endian::U64::new(n),
        }
    }

    pub fn key_from_u64(n: u64) -> <DefaultTreeFactors as Factor>::Key {
        let mut key = [0; 128];
        key[120..].copy_from_slice(&n.to_be_bytes());
        key
    }

    #[test]
    fn proptest() {
        let source = vec![];
        let mut map = BTree::<_, DefaultTreeFactors>::new(source).unwrap();
        let mut truth = HashMap::new();

        let mut rng = StdRng::seed_from_u64(31415);

        for _ in 0..100000 {
            let key = key_from_u64(rng.gen());
            let val = node(rng.gen());

            truth.insert(key, val);
            map.insert(&key, val).unwrap();
        }

        map.validate_tree().unwrap();

        for (k, v) in truth {
            assert_eq!(map.search(&k).unwrap(), Some(v));
        }
    }
}
