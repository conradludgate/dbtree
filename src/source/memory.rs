use std::io;

use crate::{array::Array, onecopy, NodeRef};

use super::{PageSink, PageSource};

impl<Page: Array<u8>> PageSource<Page> for Vec<Page> {
    fn read(&self, page_ref: NodeRef) -> io::Result<Option<Page>> {
        Ok(usize::try_from(page_ref.offset.get())
            .ok()
            .and_then(|page| self.get(page))
            .map(onecopy))
    }
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
