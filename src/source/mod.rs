use std::io;

use crate::{array::Array, checksum::update_checksum, HeapPtr};

mod memory;
mod file;

pub trait PageSource<Page: Array<u8>> {
    fn read(&self, page_ref: HeapPtr) -> io::Result<Option<Page>>;

    fn must_read(&self, page_ref: HeapPtr) -> io::Result<Page> {
        self.read(page_ref)?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "noderef does not reference a valid page",
            )
        })
    }
}

pub trait PageSink<Page: Array<u8>>: PageSource<Page> {
    fn write(&mut self, page_ref: HeapPtr, page: &Page) -> io::Result<()>;

    fn write_page(&mut self, page_ref: HeapPtr, page: &mut Page) -> io::Result<()> {
        update_checksum(page);
        self.write(page_ref, page)
    }
}
