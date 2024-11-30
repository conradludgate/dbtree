use core::fmt;
use std::{
    fmt::Debug,
    io::{self},
    marker::PhantomData,
    ops::{Bound, RangeBounds},
};

use array::Array;
use zerocopy::{
    big_endian, little_endian, FromBytes, FromZeros, Immutable, IntoBytes, KnownLayout, Unaligned,
};

/// A B+Tree that stores 128 byte keys and 4kib values
pub struct BTree<S, F: Factor> {
    page_store: S,
    factor: PhantomData<F>,
    metadata: FileMetadata,
}

struct TreeFmt<'a, S, F: Factor> {
    page_store: &'a S,
    factor: PhantomData<F>,
    node: NodeRef,
    depth: u64,
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
    type Page = [u8; 4096];
    type Key = [u8; 128];

    type Branch<T: IntoBytes + FromBytes + KnownLayout + Unaligned + Immutable> = [T; 30];
}

// const fn internal_padding<F: Factor>() -> usize {
//     const {
//         assert!(<F::Branch<F::Key> as Array::<F::Key>>::SIZE < u8::MAX as usize);

//         let keys = size_of::<F::Branch<F::Key>>();
//         let children = size_of::<F::Branch<NodeRef>>() + size_of::<NodeRef>();
//         let len = size_of::<u8>();
//         let crc = size_of::<Checksum>();

//         let total = keys + children + len + crc;
//         let page_size = F::Page::SIZE;

//         assert!(total < page_size);
//         page_size - (total)
//     }
// }

struct KeyRef<'a, F: Factor>(&'a F::Key);

impl<F: Factor> fmt::Debug for KeyRef<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Ok((rest, n)) = big_endian::U64::read_from_suffix(self.0.as_bytes()) {
            if rest.iter().all(|b| *b == 0) {
                return n.get().fmt(f);
            }
        }

        write!(f, "b\"")?;
        for &b in self.0.as_bytes() {
            // https://doc.rust-lang.org/reference/tokens.html#byte-escapes
            if b == b'\n' {
                write!(f, "\\n")?;
            } else if b == b'\r' {
                write!(f, "\\r")?;
            } else if b == b'\t' {
                write!(f, "\\t")?;
            } else if b == b'\\' || b == b'"' {
                write!(f, "\\{}", b as char)?;
            } else if b == b'\0' {
                write!(f, "\\0")?;
            // ASCII printable
            } else if (0x20..0x7f).contains(&b) {
                write!(f, "{}", b as char)?;
            } else {
                write!(f, "\\x{:02x}", b)?;
            }
        }
        write!(f, "\"")?;
        Ok(())
    }
}

impl<S: PageSource<F::Page>, F: Factor> fmt::Debug for TreeFmt<'_, S, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.node == NODE_SENTINAL {
            return f.debug_map().finish();
        }

        let page = self.page_store.must_read(self.node).unwrap();

        if self.depth > 0 {
            let node = InternalNode::<F>::ref_from_bytes(page.as_bytes())
                .expect("all 4k pages should cast exactly to an InternalNode");

            let mut f = f.debug_list();
            f.entry(&TreeFmt {
                page_store: self.page_store,
                factor: self.factor,
                node: node.min,
                depth: self.depth - 1,
            });
            for i in 0..node.len as usize {
                f.entry(&KeyRef::<F>(&node.keys[i]));
                f.entry(&TreeFmt {
                    page_store: self.page_store,
                    factor: self.factor,
                    node: node.values[i],
                    depth: self.depth - 1,
                });
            }
            f.finish()
        } else {
            let node = LeafNode::<F>::ref_from_bytes(page.as_bytes())
                .expect("all 4k pages should cast exactly to an LeafNode");
            let mut f = f.debug_map();
            for i in 0..node.len as usize {
                f.entry(&KeyRef::<F>(&node.keys[i]), &node.values[i]);
            }
            f.finish()
        }
    }
}

pub trait PageSource<Page: Array<u8>> {
    fn read(&self, page_ref: NodeRef) -> io::Result<Option<Page>>;

    fn must_read(&self, page_ref: NodeRef) -> io::Result<Page> {
        self.read(page_ref)?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "noderef does not reference a valid page",
            )
        })
    }
}

pub trait PageSink<Page: Array<u8>>: PageSource<Page> {
    fn write(&mut self, page_ref: NodeRef, page: &Page) -> io::Result<()>;

    fn write_page(&mut self, page_ref: NodeRef, page: &mut Page) -> io::Result<()> {
        update_checksum(page);
        self.write(page_ref, page)
    }
}

impl<Page: Array<u8>> PageSource<Page> for Vec<Page> {
    fn read(&self, page_ref: NodeRef) -> io::Result<Option<Page>> {
        Ok(usize::try_from(page_ref.offset.get())
            .ok()
            .and_then(|page| self.get(page))
            .map(onecopy))
    }
}

fn onecopy<T: IntoBytes + FromBytes + Immutable>(t: &T) -> T {
    T::read_from_bytes(t.as_bytes()).unwrap()
}

impl<Page: Array<u8>> PageSink<Page> for Vec<Page> {
    fn write(&mut self, page_ref: NodeRef, page: &Page) -> io::Result<()> {
        // println!("write: {page_ref:?}");

        let Ok(i) = usize::try_from(page_ref.offset.get()) else {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "page offset too large",
            ));
        };

        if i > self.len() {
            self.resize_with(i, Page::new_zeroed);
        }
        if let Some(p) = self.get_mut(i) {
            *p = onecopy(page);
        } else {
            self.push(onecopy(page));
        }

        Ok(())
    }
}

// checksum is always at the end of the page.
fn validate_page<T: FromBytes + KnownLayout + Immutable, Page: Array<u8>>(
    page: &Page,
) -> io::Result<&T> {
    let (rest, checksum) = Checksum::ref_from_suffix(page.as_bytes())
        .expect("should always be able to read a checksum from a page");

    if crc32fast::hash(rest) == checksum.get() {
        Ok(T::ref_from_prefix(rest).unwrap().0)
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "checksum integrity failure",
        ))
    }
}

// checksum is always at the end of the page.
fn validate_page_mut<T: FromBytes + IntoBytes + KnownLayout, Page: Array<u8>>(
    page: &mut Page,
) -> io::Result<&mut T> {
    let (rest, checksum) = Checksum::mut_from_suffix(page.as_mut_bytes())
        .expect("should always be able to read a checksum from a page");

    if crc32fast::hash(rest) == checksum.get() {
        Ok(T::mut_from_prefix(rest).unwrap().0)
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "checksum integrity failure",
        ))
    }
}

// checksum is always at the end of the page.
pub fn update_checksum<Page: IntoBytes + FromBytes>(page: &mut Page) {
    let (rest, checksum) = Checksum::mut_from_suffix(page.as_mut_bytes())
        .expect("should always be able to read a checksum from a page");
    checksum.set(crc32fast::hash(rest));
}

type Checksum = little_endian::U32;

pub type Page = [u8; 4096];

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

    pub fn validate_tree(&self) -> io::Result<()> {
        let depth = self.metadata.depth.get();
        let current = self.metadata.root;

        // the btree is currently empty.
        if current == NODE_SENTINAL {
            return Ok(());
        }

        self.validate_tree_bounds(current, depth, (Bound::Unbounded, Bound::Unbounded))
    }

    pub fn validate_tree_bounds(
        &self,
        node_ref: NodeRef,
        depth: u64,
        bounds: (Bound<&F::Key>, Bound<&F::Key>),
    ) -> io::Result<()> {
        let page = self.page_store.must_read(node_ref)?;

        if depth > 0 {
            let node = validate_page::<InternalNode<F>, F::Page>(&page)?;

            let len = node.len as usize;
            let keys = &node.keys[..len];

            if !keys.is_sorted() {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("invalid tree ordering: unsorted keys. {node_ref:?} depth={depth}"),
                ));
            }

            for key in keys {
                if !bounds.contains(key) {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("invalid tree ordering: key out of bounds of parent node. {node_ref:?} depth={depth}"),
                    ));
                }
            }

            if let Bound::Included(start) = bounds.0 {
                if &keys[0] != start {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("invalid tree ordering: first key not in line with parent pivot. {node_ref:?} depth={depth}"),
                    ));
                }
            }

            let values = &node.values[..len];
            self.validate_tree_bounds(node.min, depth - 1, (bounds.0, Bound::Excluded(&keys[0])))?;
            for i in 0..len {
                self.validate_tree_bounds(
                    values[i],
                    depth - 1,
                    (
                        Bound::Included(&keys[i]),
                        keys.get(i).map_or(bounds.1, Bound::Excluded),
                    ),
                )?;
            }
        } else {
            let node = validate_page::<LeafNode<F>, F::Page>(&page)?;

            let len = node.len as usize;
            let keys = &node.keys[..len];

            if !keys.is_sorted() {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("invalid tree ordering: unsorted keys. {node_ref:?} depth={depth}"),
                ));
            }

            for key in keys {
                if !bounds.contains(key) {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("invalid tree ordering: key out of bounds of parent node. {node_ref:?} depth={depth}"),
                    ));
                }
            }
        }

        Ok(())
    }

    fn search_internal_node(&self, node_ref: NodeRef, key: &F::Key) -> io::Result<NodeRef> {
        let page = self.page_store.must_read(node_ref)?;
        let node = validate_page::<InternalNode<F>, F::Page>(&page)?;

        let len = node.len as usize;
        let keys = &node.keys[..len];
        match keys.binary_search(key) {
            Err(0) => Ok(node.min),
            Ok(i) => Ok(node.values[i]),
            Err(i) => Ok(node.values[i - 1]),
        }
    }

    fn search_leaf_node(&self, node: NodeRef, key: &F::Key) -> io::Result<SearchEntry> {
        let page = self.page_store.must_read(node)?;
        let node = validate_page::<LeafNode<F>, F::Page>(&page)?;

        let len = node.len as usize;
        let keys = &node.keys[..len];
        match keys.binary_search(key) {
            Ok(i) => Ok(SearchEntry::Occupied(i, node.values[i])),
            Err(i) => Ok(SearchEntry::Vacant(i)),
        }
    }

    pub fn search(&self, key: &F::Key) -> io::Result<Option<NodeRef>> {
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
    Replaced(NodeRef),
    Split(F::Key, NodeRef),
}

enum MustInsertState {
    Inserted,
    Replaced(NodeRef),
}

enum InsertInternalState<F: Factor> {
    Inserted,
    Split(F::Key, NodeRef),
}

impl<S, F> BTree<S, F>
where
    S: PageSink<F::Page>,
    F: Factor,
{
    fn allocate(&mut self) -> io::Result<NodeRef> {
        // todo: use freelist

        if self.metadata.len == 0 {
            self.metadata.len += 1;
        }
        self.metadata.len += 1;
        let x = NodeRef {
            offset: little_endian::U64::new(self.metadata.len.get() - 1),
        };

        self.update_metadata()?;

        Ok(x)
    }

    fn update_metadata(&mut self) -> io::Result<()> {
        let mut page = F::Page::new_zeroed();
        self.metadata.write_to_prefix(page.as_mut_bytes()).unwrap();
        update_checksum(&mut page);
        self.page_store.write(NODE_SENTINAL, &page)
    }

    fn insert_leaf_node(
        &mut self,
        page_ref: NodeRef,
        key: &F::Key,
        value: NodeRef,
    ) -> io::Result<InsertState<F>> {
        let mut page = self.page_store.must_read(page_ref)?;
        let node = validate_page_mut::<LeafNode<F>, F::Page>(&mut page)?;

        let len = node.len as usize;
        let keys = &node.keys[..len];

        match keys.binary_search(key) {
            Ok(i) => {
                let old_value = std::mem::replace(&mut node.values[i], value);

                update_checksum(&mut page);
                self.page_store.write(page_ref, &page)?;

                Ok(InsertState::Replaced(old_value))
            }
            Err(i) if len < F::Branch::<u8>::SIZE => {
                node.keys[i..].rotate_right(1);
                node.values[i..].rotate_right(1);

                node.keys[i] = *key;
                node.values[i] = value;
                node.len += 1;

                update_checksum(&mut page);
                self.page_store.write(page_ref, &page)?;

                Ok(InsertState::Inserted)
            }
            // overflow
            Err(_) => {
                debug_assert_eq!(node.len, F::Branch::<u8>::SIZE as u8);

                let new_page_ref = self.allocate()?;

                let pivot;
                {
                    let mut page = F::Page::new_zeroed();
                    {
                        let (new_node, _) =
                            LeafNode::<F>::mut_from_prefix(page.as_mut_bytes()).unwrap();

                        pivot = node.split_into(new_node);
                    }

                    self.page_store.write_page(new_page_ref, &mut page)?;
                }

                update_checksum(&mut page);
                self.page_store.write(page_ref, &page)?;

                Ok(InsertState::Split(pivot, new_page_ref))
            }
        }
    }

    fn must_insert_leaf_node(
        &mut self,
        page_ref: NodeRef,
        key: &F::Key,
        value: NodeRef,
    ) -> io::Result<MustInsertState> {
        let mut page = self.page_store.must_read(page_ref)?;
        let node = validate_page_mut::<LeafNode<F>, F::Page>(&mut page)?;

        let len = node.len as usize;
        let keys = &node.keys[..len];

        match keys.binary_search(key) {
            Ok(i) => {
                let old_value = std::mem::replace(&mut node.values[i], value);

                update_checksum(&mut page);
                self.page_store.write(page_ref, &page)?;

                Ok(MustInsertState::Replaced(old_value))
            }
            Err(i) if len < F::Branch::<u8>::SIZE => {
                node.keys[i..].rotate_right(1);
                node.values[i..].rotate_right(1);

                node.keys[i] = *key;
                node.values[i] = value;
                node.len += 1;

                update_checksum(&mut page);
                self.page_store.write(page_ref, &page)?;

                Ok(MustInsertState::Inserted)
            }
            Err(_) => unreachable!("btree insert must not overflow"),
        }
    }

    fn insert_internal_node(
        &mut self,
        page_ref: NodeRef,
        key: &F::Key,
        value: NodeRef,
    ) -> io::Result<InsertInternalState<F>> {
        let mut page = self.page_store.must_read(page_ref)?;
        let node = validate_page_mut::<InternalNode<F>, F::Page>(&mut page)?;

        let len = node.len as usize;
        let keys = &node.keys[..len];
        let values = &node.values[..len];

        dbg!((keys.iter().map(KeyRef::<F>).collect::<Vec<_>>(), values));

        match keys.binary_search(key) {
            Ok(_) => unreachable!("key should not already be here"),
            Err(0) => unreachable!("when inserting, we always split off the rhs, so this should never arrive on the left"),
            Err(i) if len < F::Branch::<u8>::SIZE => {
                dbg!((KeyRef::<F>(key), i));
                node.keys[i..].rotate_right(1);
                node.values[i..].rotate_right(1);

                node.keys[i] = *key;
                node.values[i] = value;
                node.len += 1;

                update_checksum(&mut page);
                self.page_store.write(page_ref, &page)?;

                Ok(InsertInternalState::Inserted)
            }
            // overflow
            Err(_) => {
                dbg!("split internal");
                debug_assert_eq!(node.len as usize, F::Branch::<u8>::SIZE);

                let new_page_ref = self.allocate()?;

                let pivot;
                {
                    let mut page = F::Page::new_zeroed();
                    {
                        let (new_node, _) =
                            InternalNode::<F>::mut_from_prefix(page.as_mut_bytes()).unwrap();

                        pivot = node.split_into(new_node);
                    }

                    self.page_store.write_page(new_page_ref, &mut page)?;
                }

                update_checksum(&mut page);
                self.page_store.write(page_ref, &page)?;

                Ok(InsertInternalState::Split(pivot, new_page_ref))
            }
        }
    }

    // returns the old NodeRef if there was one.
    pub fn insert(&mut self, key: &F::Key, value: NodeRef) -> io::Result<Option<NodeRef>> {
        let mut depth = self.metadata.depth.get();
        let mut current = self.metadata.root;

        // the btree is currently empty.
        // allocate a new empty root.
        if current == NODE_SENTINAL {
            let root_page_ref = self.allocate()?;

            {
                let mut page = F::Page::new_zeroed();
                {
                    let (_root_node, _) =
                        LeafNode::<F>::mut_from_prefix(page.as_mut_bytes()).unwrap();
                    // zero value for the leaf is valid as empty
                }

                self.page_store.write_page(root_page_ref, &mut page)?;
            }

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

        dbg!((KeyRef::<F>(&pivot), new_node_ref));

        // the node had to split, try insert into the parent.
        let mut stack2 = vec![(*key, value)];
        current = loop {
            depth += 1;

            let Some(parent) = stack.pop() else {
                // no more parents, we need to allocate a new root.
                let root_page_ref = self.allocate()?;
                {
                    let mut page = F::Page::new_zeroed();
                    {
                        let (root_node, _) =
                            LeafNode::<F>::mut_from_prefix(page.as_mut_bytes()).unwrap();

                        root_node.len = 1;
                        root_node.keys[0] = pivot;
                        root_node.values[0] = self.metadata.root;
                        root_node.values[1] = new_node_ref;
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
                    stack2.push((pivot, new_node_ref));
                    (pivot, new_node_ref) = (p, n)
                }
            };
        };

        // walk back down the new set of internal nodes
        drop(stack);
        loop {
            let Some((key, value)) = stack2.pop() else {
                break Ok(None);
            };

            let mut depth = depth;
            let mut current = current;

            while depth > 0 {
                current = self.search_internal_node(current, &key)?;
                depth -= 1;
            }

            // insert into the leaf that now is guaranteed to have space
            self.must_insert_leaf_node(current, &key, value)?;
        }
    }
}

pub enum SearchEntry {
    Occupied(usize, NodeRef),
    Vacant(usize),
}

// pub type Key = [u8; 128];

pub fn key_from_u64(n: u64) -> <DefaultTreeFactors as Factor>::Key {
    let mut key = [0; 128];
    key[120..].copy_from_slice(&n.to_be_bytes());
    key
}

// ======== NOTES =======
// internal nodes need to store a length, each key and each node ref
// with 4KiB pages, 128 byte keys and 8 byte refs, and 1 byte length
// we could store at most 30 keys per node.
// ======================
// const <DefaultTreeFactors as Factor>::Branch::<u8>::SIZE: usize = 30;

#[derive(KnownLayout, IntoBytes, Unaligned, FromBytes, Immutable)]
#[repr(C)]
pub struct InternalNode<F: Factor> {
    keys: F::Branch<F::Key>,
    min: NodeRef,
    values: F::Branch<NodeRef>,

    len: u8,
    // _padding: [u8; 3],

    // crc: Checksum,
}

impl<F: Factor> InternalNode<F> {
    fn split_into(&mut self, rhs: &mut Self) -> F::Key {
        debug_assert_eq!(rhs.len, 0);
        debug_assert_eq!(self.len as usize, F::Branch::<u8>::SIZE);

        // orig: 30 keys, 31 values
        // ->
        // orig: 15 keys, 16 values
        // pivot: 1 key
        // new: 14 keys, 15 values

        self.len = (<DefaultTreeFactors as Factor>::Branch::<u8>::SIZE / 2) as u8;
        rhs.len = <DefaultTreeFactors as Factor>::Branch::<u8>::SIZE as u8 - self.len - 1;

        let (ka, kb) = self.keys.as_ref().split_at(self.len as usize);
        let (kp, kb) = kb.split_first().unwrap();
        debug_assert_eq!(ka.len(), self.len as usize);
        debug_assert_eq!(kb.len(), rhs.len as usize);

        let (va, vb) = self.values.as_ref().split_at(self.len as usize);
        let (vp, vb) = vb.split_first().unwrap();
        debug_assert_eq!(va.len(), self.len as usize);
        debug_assert_eq!(vb.len(), rhs.len as usize);

        rhs.keys[..rhs.len as usize].copy_from_slice(kb);
        rhs.min = *vp;
        rhs.values[..rhs.len as usize].copy_from_slice(vb);

        *kp
    }
}

#[derive(KnownLayout, IntoBytes, Unaligned, FromBytes, Immutable)]
#[repr(C)]
pub struct LeafNode<F: Factor> {
    keys: F::Branch<F::Key>,
    values: F::Branch<NodeRef>,

    // /// represents a doubly linked list of leaf nodes.
    // /// we only had space for 1 pointer, so this is an xor link.
    // xor_link: NodeRef,
    len: u8,
    // _padding: [u8; 3],

    // crc: Checksum,
}

impl<F: Factor> LeafNode<F> {
    fn split_into(&mut self, rhs: &mut Self) -> F::Key {
        debug_assert_eq!(rhs.len, 0);
        debug_assert_eq!(
            self.len,
            <DefaultTreeFactors as Factor>::Branch::<u8>::SIZE as u8
        );

        // orig: 30 keys, 30 values
        // ->
        // orig: 15 keys, 15 values
        // new: 15 keys, 15 values

        self.len = (<DefaultTreeFactors as Factor>::Branch::<u8>::SIZE / 2) as u8;
        rhs.len = <DefaultTreeFactors as Factor>::Branch::<u8>::SIZE as u8 - self.len;

        dbg!(self.len, rhs.len);

        let keys_pivot = self.keys[self.len as usize];
        let keys_r = &self.keys[self.len as usize..];
        let values_r = &self.values[self.len as usize..];

        rhs.keys[..rhs.len as usize].copy_from_slice(keys_r);
        rhs.values[..rhs.len as usize].copy_from_slice(values_r);

        keys_pivot
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
    root: NodeRef,
    depth: little_endian::U64,

    // a doubly linked list for the free pages
    free_head: NodeRef,
    free_tail: NodeRef,
}

/// Reserved for the file metadata.
/// In any other position it is a "None" value.
/// Can be used to indicate that no entry is present.
const NODE_SENTINAL: NodeRef = NodeRef {
    offset: little_endian::U64::new(0),
};

#[derive(KnownLayout, IntoBytes, Unaligned, Immutable, FromBytes, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct NodeRef {
    offset: little_endian::U64,
}

impl Debug for NodeRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.offset.get().fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rand::{rngs::StdRng, Rng, SeedableRng};
    use zerocopy::little_endian;

    use crate::{key_from_u64, BTree, DefaultTreeFactors, TreeFmt};

    #[test]
    fn check() {
        let source = vec![];
        let mut map = BTree::<_, DefaultTreeFactors>::new(source).unwrap();

        for i in 1..=481 {
            map.insert(
                &key_from_u64(i),
                crate::NodeRef {
                    offset: little_endian::U64::new(i),
                },
            )
            .unwrap();
        }

        let tree = TreeFmt {
            page_store: &map.page_store,
            factor: map.factor,
            node: map.metadata.root,
            depth: map.metadata.depth.get(),
        };
        dbg!(tree);
    }

    #[test]
    fn proptest() {
        let source = vec![];
        let mut map = BTree::<_, DefaultTreeFactors>::new(source).unwrap();
        let mut truth = HashMap::new();

        let mut rng = StdRng::seed_from_u64(31415);

        for _ in 1..=100 {
            let key = key_from_u64(rng.gen());
            let val = crate::NodeRef {
                offset: little_endian::U64::new(rng.gen()),
            };

            truth.insert(key, val);
            map.insert(&key, val).unwrap();
            map.validate_tree().unwrap();
        }

        for (k, v) in truth {
            assert_eq!(map.search(&k).unwrap(), Some(v));
        }
    }
}
