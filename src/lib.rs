use core::fmt;
use std::{
    fmt::Debug,
    io::{self},
    ops::{Bound, RangeBounds},
};

use zerocopy::{
    little_endian, FromBytes, FromZeros, Immutable, IntoBytes, KnownLayout, Unaligned, U64,
};

/// A B+Tree that stores 128 byte keys and 4kib values
pub struct BTree<S> {
    page_store: S,
    metadata: FileMetadata,
}

struct TreeFmt<'a, S> {
    page_store: &'a S,
    node: NodeRef,
    depth: u64,
}

struct KeyRef<'a>(&'a Key);

impl fmt::Debug for KeyRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0[..120] == [0; 120] {
            return u64::from_be_bytes(self.0[120..].try_into().unwrap()).fmt(f);
        }
        write!(f, "b\"")?;
        for &b in self.0 {
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

impl<'a, S: PageSource> fmt::Debug for TreeFmt<'a, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.node == NODE_SENTINAL {
            return f.debug_map().finish();
        }

        let page = self.page_store.must_read(self.node).unwrap();

        if self.depth > 0 {
            let node = InternalNode::ref_from_bytes(&page)
                .expect("all 4k pages should cast exactly to an InternalNode");

            let mut f = f.debug_list();
            for i in 0..node.len as usize + 1 {
                f.entry(&TreeFmt {
                    page_store: self.page_store,
                    node: node.values[i],
                    depth: self.depth - 1,
                });
                if i < node.len as usize {
                    f.entry(&KeyRef(&node.keys[i]));
                }
            }
            f.finish()
        } else {
            let node = LeafNode::ref_from_bytes(&page)
                .expect("all 4k pages should cast exactly to an LeafNode");
            let mut f = f.debug_map();
            for i in 0..node.len as usize {
                f.entry(&KeyRef(&node.keys[i]), &node.values[i]);
            }
            f.finish()
        }
    }
}

pub trait PageSource {
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

pub trait PageSink: PageSource {
    fn write(&mut self, page_ref: NodeRef, page: &Page) -> io::Result<()>;

    fn write_page(&mut self, page_ref: NodeRef, page: &mut Page) -> io::Result<()> {
        update_checksum(page);
        self.write(page_ref, page)
    }
}

impl PageSource for Vec<Page> {
    fn read(&self, page_ref: NodeRef) -> io::Result<Option<Page>> {
        Ok(usize::try_from(page_ref.offset.get())
            .ok()
            .and_then(|page| self.get(page))
            .copied())
    }
}

impl PageSink for Vec<Page> {
    fn write(&mut self, page_ref: NodeRef, page: &Page) -> io::Result<()> {
        // println!("write: {page_ref:?}");

        let Ok(i) = usize::try_from(page_ref.offset.get()) else {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "page offset too large",
            ));
        };

        if i > self.len() {
            self.resize_with(i, || [0; 4096]);
        }
        if let Some(p) = self.get_mut(i) {
            *p = *page;
        } else {
            self.push(*page);
        }

        Ok(())
    }
}

// checksum is always at the end of the page.
fn validate_page(page: &Page) -> io::Result<()> {
    let (rest, checksum) = Checksum::read_from_suffix(page)
        .expect("should always be able to read a checksum from a page");

    if crc32fast::hash(rest) == checksum.get() {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "checksum integrity failure",
        ))
    }
}

// checksum is always at the end of the page.
pub fn update_checksum(page: &mut Page) {
    let (rest, checksum) = Checksum::mut_from_suffix(page)
        .expect("should always be able to read a checksum from a page");
    checksum.set(crc32fast::hash(rest));
}

type Checksum = little_endian::U32;

pub type Page = [u8; 4096];

impl<S: PageSource> BTree<S> {
    pub fn new(s: S) -> io::Result<Self> {
        let metadata;
        if let Some(page) = s.read(NODE_SENTINAL)? {
            validate_page(&page)?;

            let (m, _rest) = FileMetadata::ref_from_prefix(&page)
                .expect("all 4k pages should cast to a FileMetadata");
            metadata = *m;
        } else {
            // zero value represents an empty tree.
            metadata = FileMetadata::new_zeroed();
        }

        Ok(Self {
            page_store: s,
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
        bounds: (Bound<&Key>, Bound<&Key>),
    ) -> io::Result<()> {
        let page = self.page_store.must_read(node_ref)?;
        validate_page(&page)?;

        if depth > 0 {
            let node = InternalNode::ref_from_bytes(&page)
                .expect("all 4k pages should cast exactly to an InternalNode");

            let len = node.len as usize;
            let keys = &node.keys[..len];
            let values = &node.values[..len + 1];

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

            self.validate_tree_bounds(values[0], depth - 1, (bounds.0, Bound::Excluded(&keys[0])))?;
            for i in 1..=len {
                self.validate_tree_bounds(
                    values[i],
                    depth - 1,
                    (
                        Bound::Included(&keys[i - 1]),
                        keys.get(i).map_or(bounds.1, Bound::Excluded),
                    ),
                )?;
            }
        } else {
            let node = LeafNode::ref_from_bytes(&page)
                .expect("all 4k pages should cast exactly to an LeafNode");

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

    fn search_internal_node(&self, node_ref: NodeRef, key: &Key) -> io::Result<NodeRef> {
        let page = self.page_store.must_read(node_ref)?;
        validate_page(&page)?;

        let node = InternalNode::ref_from_bytes(&page)
            .expect("all 4k pages should cast exactly to an InternalNode");
        let len = node.len as usize;
        let keys = &node.keys[..len];
        match keys.binary_search(key) {
            Ok(i) => Ok(node.values[i + 1]),
            Err(i) => Ok(node.values[i]),
        }
    }

    fn search_leaf_node(&self, node: NodeRef, key: &Key) -> io::Result<SearchEntry> {
        let page = self.page_store.must_read(node)?;
        validate_page(&page)?;

        let node = LeafNode::ref_from_bytes(&page)
            .expect("all 4k pages should cast exactly to a LeafNode");

        let len = node.len as usize;
        let keys = &node.keys[..len];
        match keys.binary_search(key) {
            Ok(i) => Ok(SearchEntry::Occupied(i, node.values[i])),
            Err(i) => Ok(SearchEntry::Vacant(i)),
        }
    }

    pub fn search(&self, key: &Key) -> io::Result<Option<NodeRef>> {
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

enum InsertState {
    Inserted,
    Replaced(NodeRef),
    Split(Key, NodeRef),
}

enum MustInsertState {
    Inserted,
    Replaced(NodeRef),
}

enum InsertInternalState {
    Inserted,
    Split(Key, NodeRef),
}

impl<S: PageSink> BTree<S> {
    fn allocate(&mut self) -> io::Result<NodeRef> {
        // todo: use freelist

        if self.metadata.len == 0 {
            self.metadata.len += 1;
        }
        self.metadata.len += 1;
        let x = NodeRef {
            offset: U64::new(self.metadata.len.get() - 1),
        };

        self.update_metadata()?;

        Ok(x)
    }

    fn update_metadata(&mut self) -> io::Result<()> {
        let mut page = [0; 4096];
        self.metadata.write_to_prefix(&mut page).unwrap();
        update_checksum(&mut page);
        self.page_store.write(NODE_SENTINAL, &page)
    }

    fn insert_leaf_node(
        &mut self,
        page_ref: NodeRef,
        key: &Key,
        value: NodeRef,
    ) -> io::Result<InsertState> {
        let mut page = self.page_store.must_read(page_ref)?;
        validate_page(&page)?;

        let node = LeafNode::mut_from_bytes(&mut page)
            .expect("all 4k pages should cast exactly to a LeafNode");

        let len = node.len as usize;
        let keys = &node.keys[..len];

        match keys.binary_search(key) {
            Ok(i) => {
                let old_value = std::mem::replace(&mut node.values[i], value);

                update_checksum(&mut page);
                self.page_store.write(page_ref, &page)?;

                Ok(InsertState::Replaced(old_value))
            }
            Err(i) if len < BRANCH_FACTOR => {
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
                debug_assert_eq!(node.len, BRANCH_FACTOR as u8);

                let new_page_ref = self.allocate()?;
                let mut new_node = LeafNode::new_zeroed();

                let pivot = node.split_into(&mut new_node);

                let new_page = zerocopy::transmute_mut!(&mut new_node);
                update_checksum(new_page);
                self.page_store.write(new_page_ref, new_page)?;

                update_checksum(&mut page);
                self.page_store.write(page_ref, &page)?;

                Ok(InsertState::Split(pivot, new_page_ref))
            }
        }
    }

    fn must_insert_leaf_node(
        &mut self,
        page_ref: NodeRef,
        key: &Key,
        value: NodeRef,
    ) -> io::Result<MustInsertState> {
        let mut page = self.page_store.must_read(page_ref)?;
        validate_page(&page)?;

        let node = LeafNode::mut_from_bytes(&mut page)
            .expect("all 4k pages should cast exactly to a LeafNode");

        let len = node.len as usize;
        let keys = &node.keys[..len];

        match keys.binary_search(key) {
            Ok(i) => {
                let old_value = std::mem::replace(&mut node.values[i], value);

                update_checksum(&mut page);
                self.page_store.write(page_ref, &page)?;

                Ok(MustInsertState::Replaced(old_value))
            }
            Err(i) if len < BRANCH_FACTOR => {
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
        key: &Key,
        value: NodeRef,
    ) -> io::Result<InsertInternalState> {
        let mut page = self.page_store.must_read(page_ref)?;
        validate_page(&page)?;

        let node = InternalNode::mut_from_bytes(&mut page)
            .expect("all 4k pages should cast exactly to a InternalNode");

        let len = node.len as usize;
        let keys = &node.keys[..len];
        let values = &node.values[..len + 1];

        dbg!((keys.iter().map(KeyRef).collect::<Vec<_>>(), values));

        match keys.binary_search(key) {
            Ok(_) => unreachable!("key should not already be here"),
            Err(0) => unreachable!("when inserting, we always split off the rhs, so this should never arrive on the left"),
            Err(i) if len < BRANCH_FACTOR => {
                dbg!((KeyRef(key), i));
                node.keys[i..].rotate_right(1);
                node.values[i + 1..].rotate_right(1);

                node.keys[i] = *key;
                node.values[i + 1] = value;
                node.len += 1;

                update_checksum(&mut page);
                self.page_store.write(page_ref, &page)?;

                Ok(InsertInternalState::Inserted)
            }
            // overflow
            Err(_) => {
                dbg!("split internal");
                debug_assert_eq!(node.len, BRANCH_FACTOR as u8);

                let new_page_ref = self.allocate()?;
                let mut new_node = InternalNode::new_zeroed();

                let pivot = node.split_into(&mut new_node);

                let new_page = zerocopy::transmute_mut!(&mut new_node);
                update_checksum(new_page);
                self.page_store.write(new_page_ref, new_page)?;

                update_checksum(&mut page);
                self.page_store.write(page_ref, &page)?;

                Ok(InsertInternalState::Split(pivot, new_page_ref))
            }
        }
    }

    // returns the old NodeRef if there was one.
    pub fn insert(&mut self, key: &Key, value: NodeRef) -> io::Result<Option<NodeRef>> {
        let mut depth = self.metadata.depth.get();
        let mut current = self.metadata.root;

        // the btree is currently empty.
        // allocate a new empty root.
        if current == NODE_SENTINAL {
            let root_page_ref = self.allocate()?;

            let mut root_node = LeafNode::new_zeroed();

            self.page_store
                .write_page(root_page_ref, zerocopy::transmute_mut!(&mut root_node))?;

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

        dbg!((KeyRef(&pivot), new_node_ref));

        // the node had to split, try insert into the parent.
        current = loop {
            depth += 1;

            let Some(parent) = stack.pop() else {
                // no more parents, we need to allocate a new root.
                let root_page_ref = self.allocate()?;

                let mut root_node = LeafNode::new_zeroed();

                root_node.len = 1;
                root_node.keys[0] = pivot;
                root_node.values[0] = self.metadata.root;
                root_node.values[1] = new_node_ref;

                self.page_store
                    .write_page(root_page_ref, zerocopy::transmute_mut!(&mut root_node))?;

                self.metadata.root = root_page_ref;
                self.metadata.depth += 1;
                self.update_metadata()?;

                break root_page_ref;
            };

            // insert into the parent, this might split again
            match self.insert_internal_node(parent, &pivot, new_node_ref)? {
                InsertInternalState::Inserted => break parent,
                InsertInternalState::Split(p, n) => (pivot, new_node_ref) = (p, n),
            };
        };

        // walk back down the new set of internal nodes
        drop(stack);
        while depth > 0 {
            current = self.search_internal_node(current, key)?;
            depth -= 1;
        }

        // insert into the leaf that now is guaranteed to have space
        match self.must_insert_leaf_node(current, key, value)? {
            MustInsertState::Inserted => Ok(None),
            MustInsertState::Replaced(node_ref) => Ok(Some(node_ref)),
        }
    }
}

pub enum SearchEntry {
    Occupied(usize, NodeRef),
    Vacant(usize),
}

pub type Key = [u8; 128];

pub fn key_from_u64(n: u64) -> Key {
    let mut key = [0; 128];
    key[120..].copy_from_slice(&n.to_be_bytes());
    key
}

// ======== NOTES =======
// internal nodes need to store a length, each key and each node ref
// with 4KiB pages, 128 byte keys and 8 byte refs, and 1 byte length
// we could store at most 30 keys per node.
// ======================
const BRANCH_FACTOR: usize = 30;

#[derive(KnownLayout, IntoBytes, Unaligned, FromBytes, Immutable)]
#[repr(C)]
pub struct InternalNode {
    keys: [Key; BRANCH_FACTOR],
    values: [NodeRef; BRANCH_FACTOR + 1],

    len: u8,
    _padding: [u8; 3],

    crc: Checksum,
}

impl InternalNode {
    fn split_into(&mut self, rhs: &mut Self) -> Key {
        debug_assert_eq!(rhs.len, 0);
        debug_assert_eq!(self.len, BRANCH_FACTOR as u8);

        // orig: 30 keys, 31 values
        // ->
        // orig: 15 keys, 16 values
        // pivot: 1 key
        // new: 14 keys, 15 values

        self.len = (BRANCH_FACTOR / 2) as u8;
        rhs.len = BRANCH_FACTOR as u8 - self.len - 1;

        let keys_pivot = self.keys[self.len as usize];
        let keys_r = &self.keys[self.len as usize + 1..];
        let values_r = &self.values[self.len as usize + 1..];

        rhs.keys[..rhs.len as usize].copy_from_slice(keys_r);
        rhs.values[..rhs.len as usize + 1].copy_from_slice(values_r);

        keys_pivot
    }
}

#[derive(KnownLayout, IntoBytes, Unaligned, FromBytes, Immutable)]
#[repr(C)]
pub struct LeafNode {
    keys: [Key; BRANCH_FACTOR],
    values: [NodeRef; BRANCH_FACTOR],

    /// represents a doubly linked list of leaf nodes.
    /// we only had space for 1 pointer, so this is an xor link.
    xor_link: NodeRef,

    len: u8,
    _padding: [u8; 3],

    crc: Checksum,
}

impl LeafNode {
    fn split_into(&mut self, rhs: &mut Self) -> Key {
        debug_assert_eq!(rhs.len, 0);
        debug_assert_eq!(self.len, BRANCH_FACTOR as u8);

        // orig: 30 keys, 30 values
        // ->
        // orig: 15 keys, 15 values
        // new: 15 keys, 15 values

        self.len = (BRANCH_FACTOR / 2) as u8;
        rhs.len = BRANCH_FACTOR as u8 - self.len;

        dbg!(self.len, rhs.len);

        let keys_pivot = self.keys[self.len as usize];
        let keys_r = &self.keys[self.len as usize..];
        let values_r = &self.values[self.len as usize..];

        rhs.keys[..rhs.len as usize].copy_from_slice(keys_r);
        rhs.values[..rhs.len as usize].copy_from_slice(values_r);

        keys_pivot
    }

    fn next_leaf(&self, prev_ref: NodeRef) -> NodeRef {
        NodeRef {
            offset: little_endian::U64::new(self.xor_link.offset.get() ^ prev_ref.offset.get()),
        }
    }

    fn prev_leaf(&self, next_ref: NodeRef) -> NodeRef {
        NodeRef {
            offset: little_endian::U64::new(self.xor_link.offset.get() ^ next_ref.offset.get()),
        }
    }
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

    use crate::{key_from_u64, BTree, TreeFmt};

    #[test]
    fn check() {
        let source = vec![];
        let mut map = BTree::new(source).unwrap();

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
            node: map.metadata.root,
            depth: map.metadata.depth.get(),
        };
        dbg!(tree);
    }

    #[test]
    fn proptest() {
        let source = vec![];
        let mut map = BTree::new(source).unwrap();
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
