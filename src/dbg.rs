use core::fmt;
use std::{
    io::{self},
    marker::PhantomData,
    ops::{Bound, RangeBounds},
};

use zerocopy::{big_endian, FromBytes, IntoBytes};

use crate::{
    checksum::validate_page, source::PageSource, BTree, Factor, HeapPtr, InternalNode, InternalPtr,
    LeafNode, LeafPtr, NodePtr, NODE_SENTINAL,
};

#[allow(dead_code)]
struct TreeFmt<'a, S, F: Factor> {
    page_store: &'a S,
    factor: PhantomData<F>,
    node: NodePtr,
    depth: u64,
}

impl<S: PageSource<F::Page>, F: Factor> fmt::Debug for TreeFmt<'_, S, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.node.0 == NODE_SENTINAL {
            return f.debug_map().finish();
        }

        let page = self.page_store.must_read(self.node.0).unwrap();

        if self.depth > 0 {
            let (node, _) = InternalNode::<F>::ref_from_prefix(page.as_bytes())
                .expect("all 4k pages should cast exactly to an InternalNode");

            let mut f = f.debug_list();
            f.entry(&TreeFmt {
                page_store: self.page_store,
                factor: self.factor,
                node: node.min,
                depth: self.depth - 1,
            });
            for i in 0..node.len.get() as usize {
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
            let (node, _) = LeafNode::<F>::ref_from_prefix(page.as_bytes())
                .expect("all 4k pages should cast exactly to an LeafNode");
            let mut f = f.debug_map();
            for i in 0..node.len.get() as usize {
                f.entry(&KeyRef::<F>(&node.keys[i]), &node.values[i]);
            }
            f.finish()
        }
    }
}

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

impl<S: PageSource<F::Page>, F: Factor> BTree<S, F> {
    pub fn validate_tree(&self) -> io::Result<()> {
        let depth = self.metadata.depth.get();
        let current = self.metadata.root;

        // the btree is currently empty.
        if current.0 == NODE_SENTINAL {
            return Ok(());
        }

        self.validate_tree_bounds(current, depth, (Bound::Unbounded, Bound::Unbounded))
    }

    pub fn validate_tree_bounds(
        &self,
        node_ref: NodePtr,
        depth: u64,
        bounds: (Bound<&F::Key>, Bound<&F::Key>),
    ) -> io::Result<()> {
        let page = self.page_store.must_read(node_ref.0)?;

        if depth > 0 {
            let node = validate_page::<InternalNode<F>, F::Page>(&page)?;

            let len = node.len.get() as usize;
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
                        format!("invalid tree ordering: key {:?} out of bounds {:?} of parent node. {node_ref:?} depth={depth}",
                        KeyRef::<F>(key),
                        (bounds.0.map(KeyRef::<F>),bounds.1.map(KeyRef::<F>))
                    ),
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
                        keys.get(i + 1).map_or(bounds.1, Bound::Excluded),
                    ),
                )?;
            }
        } else {
            let node = validate_page::<LeafNode<F>, F::Page>(&page)?;

            let len = node.len.get() as usize;
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
                        format!("invalid tree ordering: key {:?} out of bounds {:?} of parent node. {node_ref:?} depth={depth}",
                        KeyRef::<F>(key),
                        (bounds.0.map(KeyRef::<F>),bounds.1.map(KeyRef::<F>))),
                    ));
                }
            }

            if let Bound::Included(start) = bounds.0 {
                if &keys[0] != start {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!(
                            "invalid tree ordering: first key {:?} not in line with parent pivot {:?}. {node_ref:?} depth={depth}",
                            KeyRef::<F>(&keys[0]),
                            KeyRef::<F>(start),
                        ),
                    ));
                }
            }
        }

        Ok(())
    }
}

impl fmt::Debug for HeapPtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.offset.get().fmt(f)
    }
}

impl fmt::Debug for NodePtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl fmt::Debug for LeafPtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.private.fmt(f)
    }
}

impl fmt::Debug for InternalPtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.private.fmt(f)
    }
}
