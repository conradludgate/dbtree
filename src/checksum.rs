use std::io;

use zerocopy::{little_endian, FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::array::Array;

// checksum is always at the end of the page.
pub(crate) fn validate_page<T: FromBytes + KnownLayout + Immutable, Page: Array<u8>>(
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
pub(crate) fn validate_page_mut<T: FromBytes + IntoBytes + KnownLayout, Page: Array<u8>>(
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
pub(crate) fn update_checksum<Page: IntoBytes + FromBytes>(page: &mut Page) {
    let (rest, checksum) = Checksum::mut_from_suffix(page.as_mut_bytes())
        .expect("should always be able to read a checksum from a page");
    checksum.set(crc32fast::hash(rest));
}

type Checksum = little_endian::U32;
