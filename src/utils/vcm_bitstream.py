"""Simple sequence container for actual DMC P-frame bitstreams."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator


MAGIC = b"VCM1"
SEQUENCE_HEADER = struct.Struct(">4sHHfIB")
FRAME_HEADER = struct.Struct(">BI")
FLAG_EXTERNAL_SEED = 1
FLAG_TWO_ENTROPY_CODERS = 2


@dataclass(frozen=True)
class SequenceHeader:
    width: int
    height: int
    fps: float
    coded_frames: int
    external_seed: bool
    two_entropy_coders: bool


@dataclass(frozen=True)
class FramePacket:
    qp: int
    bitstream: bytes


class VCMSequenceWriter:
    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        fps: float,
        coded_frames: int,
        external_seed: bool = True,
        two_entropy_coders: bool = False,
    ):
        if not 0 < width < 65536 or not 0 < height < 65536:
            raise ValueError("Sequence dimensions must fit unsigned 16-bit fields")
        if fps <= 0 or coded_frames <= 0:
            raise ValueError("fps and coded_frames must be positive")

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file: BinaryIO = self.path.open("wb")
        self.expected_frames = int(coded_frames)
        self.written_frames = 0
        flags = (
            (FLAG_EXTERNAL_SEED if external_seed else 0)
            | (FLAG_TWO_ENTROPY_CODERS if two_entropy_coders else 0)
        )
        self.file.write(
            SEQUENCE_HEADER.pack(
                MAGIC,
                int(width),
                int(height),
                float(fps),
                int(coded_frames),
                flags,
            )
        )

    def write_frame(self, qp: int, bitstream: bytes) -> None:
        if not 0 <= qp <= 255:
            raise ValueError("qp must fit an unsigned byte")
        if self.written_frames >= self.expected_frames:
            raise RuntimeError("More frames were written than declared in the sequence header")
        self.file.write(FRAME_HEADER.pack(int(qp), len(bitstream)))
        self.file.write(bitstream)
        self.written_frames += 1

    def close(self) -> None:
        if self.file.closed:
            return
        self.file.close()
        if self.written_frames != self.expected_frames:
            raise RuntimeError(
                f"Container declares {self.expected_frames} frames but "
                f"{self.written_frames} were written"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            self.file.close()


class VCMSequenceReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.file: BinaryIO = self.path.open("rb")
        raw_header = self.file.read(SEQUENCE_HEADER.size)
        if len(raw_header) != SEQUENCE_HEADER.size:
            raise ValueError(f"Truncated sequence header: {self.path}")
        magic, width, height, fps, coded_frames, flags = SEQUENCE_HEADER.unpack(raw_header)
        if magic != MAGIC:
            raise ValueError(f"Invalid VCM bitstream magic in {self.path}")
        self.header = SequenceHeader(
            width=width,
            height=height,
            fps=fps,
            coded_frames=coded_frames,
            external_seed=bool(flags & FLAG_EXTERNAL_SEED),
            two_entropy_coders=bool(flags & FLAG_TWO_ENTROPY_CODERS),
        )

    def frames(self) -> Iterator[FramePacket]:
        for _ in range(self.header.coded_frames):
            raw_header = self.file.read(FRAME_HEADER.size)
            if len(raw_header) != FRAME_HEADER.size:
                raise ValueError(f"Truncated frame header in {self.path}")
            qp, length = FRAME_HEADER.unpack(raw_header)
            bitstream = self.file.read(length)
            if len(bitstream) != length:
                raise ValueError(f"Truncated frame payload in {self.path}")
            yield FramePacket(qp=qp, bitstream=bitstream)
        if self.file.read(1):
            raise ValueError(f"Unexpected trailing bytes in {self.path}")

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
