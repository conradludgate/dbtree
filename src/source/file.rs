use std::{
    fs::File,
    io::{self, Read, Seek, Write},
};

use crate::{array::Array, HeapPtr};

use super::{PageSink, PageSource};

impl<Page: Array<u8>> PageSource<Page> for File {
    fn read(&self, page_ref: HeapPtr) -> io::Result<Option<Page>> {
        <&File as PageSource<Page>>::read(&self, page_ref)
    }
}

impl<Page: Array<u8>> PageSink<Page> for File {
    fn write(&mut self, page_ref: HeapPtr, page: &Page) -> io::Result<()> {
        <&File as PageSink<Page>>::write(&mut &*self, page_ref, page)
    }
}

impl<Page: Array<u8>> PageSource<Page> for &File {
    fn read(&self, page_ref: HeapPtr) -> io::Result<Option<Page>> {
        let offset = page_ref.offset.get();

        if self.metadata()?.len() <= offset {
            return Ok(None);
        }

        let mut page = Page::new_zeroed();

        let mut this = *self;
        this.seek(io::SeekFrom::Start(offset))?;
        this.read_exact(page.as_mut_bytes())?;

        Ok(Some(page))
    }
}

impl<Page: Array<u8>> PageSink<Page> for &File {
    fn write(&mut self, page_ref: HeapPtr, page: &Page) -> io::Result<()> {
        let mut this = *self;
        this.seek(io::SeekFrom::Start(page_ref.offset.get()))?;
        this.write_all(page.as_bytes())?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rand::{rngs::StdRng, Rng, SeedableRng};
    use tempfile::NamedTempFile;
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
        let source = NamedTempFile::new().unwrap();
        let mut truth = HashMap::new();

        {
            let mut map = BTree::<_, DefaultTreeFactors>::new(source.as_file()).unwrap();

            let mut rng = StdRng::seed_from_u64(31415);

            for _ in 0..100000 {
                let key = key_from_u64(rng.gen());
                let val = node(rng.gen());

                truth.insert(key, val);
                map.insert(&key, val).unwrap();
            }

            map.validate_tree().unwrap();

            for (k, v) in truth.iter() {
                assert_eq!(map.search(k).unwrap().as_ref(), Some(v));
            }
        }

        let map = BTree::<_, DefaultTreeFactors>::new(source.as_file()).unwrap();
        map.validate_tree().unwrap();
        for (k, v) in truth {
            assert_eq!(map.search(&k).unwrap(), Some(v));
        }
    }
}
