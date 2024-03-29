# http://blog.notdot.net/2009/11/Damn-Cool-Algorithms-Spatial-indexing-with-Quadtrees-and-Hilbert-Curves
# pymorton (https://github.com/trevorprater/pymorton)
# Author: trevor.prater@gmail.com
# License: MIT

import sys

_DIVISORS = [180.0 / 2 ** n for n in range(32)]


def __part1by1_32(n):
    n = n & 0x0000ffff                  # base10: 65535,      binary: 1111111111111111,                 len: 16
    n = (n | (n << 8))  & 0x00FF00FF # base10: 16711935,   binary: 111111110000000011111111,         len: 24
    n = (n | (n << 4))  & 0x0F0F0F0F # base10: 252645135,  binary: 1111000011110000111100001111,     len: 28
    n = (n | (n << 2))  & 0x33333333 # base10: 858993459,  binary: 110011001100110011001100110011,   len: 30
    n = (n | (n << 1))  & 0x55555555 # base10: 1431655765, binary: 1010101010101010101010101010101,  len: 31

    return n


def __part1by2_32(n):
    n = n & 0x000003ff                  # base10: 1023,       binary: 1111111111,                       len: 10
    n = (n ^ (n << 16)) & 0xff0000ff # base10: 4278190335, binary: 11111111000000000000000011111111, len: 32
    n = (n ^ (n << 8))  & 0x0300f00f # base10: 50393103,   binary: 11000000001111000000001111,       len: 26
    n = (n ^ (n << 4))  & 0x030c30c3 # base10: 51130563,   binary: 11000011000011000011000011,       len: 26
    n = (n ^ (n << 2))  & 0x09249249 # base10: 153391689,  binary: 1001001001001001001001001001,     len: 28

    return n


def __unpart1by1_32(n):
    n = n & 0x55555555                  # base10: 1431655765, binary: 1010101010101010101010101010101,  len: 31
    n = (n ^ (n >> 1))  & 0x33333333 # base10: 858993459,  binary: 110011001100110011001100110011,   len: 30
    n = (n ^ (n >> 2))  & 0x0f0f0f0f # base10: 252645135,  binary: 1111000011110000111100001111,     len: 28
    n = (n ^ (n >> 4))  & 0x00ff00ff # base10: 16711935,   binary: 111111110000000011111111,         len: 24
    n = (n ^ (n >> 8))  & 0x0000ffff # base10: 65535,      binary: 1111111111111111,                 len: 16

    return n


def __unpart1by2_32(n):
    n = n & 0x09249249                  # base10: 153391689,  binary: 1001001001001001001001001001,     len: 28
    n = (n ^ (n >> 2))  & 0x030c30c3 # base10: 51130563,   binary: 11000011000011000011000011,       len: 26
    n = (n ^ (n >> 4))  & 0x0300f00f # base10: 50393103,   binary: 11000000001111000000001111,       len: 26
    n = (n ^ (n >> 8))  & 0xff0000ff # base10: 4278190335, binary: 11111111000000000000000011111111, len: 32
    n = (n ^ (n >> 16)) & 0x000003ff # base10: 1023,       binary: 1111111111,                       len: 10

    return n


def __part1by1_64(n):
    n = n & 0x00000000ffffffff                  # binary: 11111111111111111111111111111111,                                len: 32
    n = (n | (n << 16)) & 0x0000FFFF0000FFFF # binary: 1111111111111111000000001111111111111111,                        len: 40
    n = (n | (n << 8))  & 0x00FF00FF00FF00FF # binary: 11111111000000001111111100000000111111110000000011111111,        len: 56
    n = (n | (n << 4))  & 0x0F0F0F0F0F0F0F0F # binary: 111100001111000011110000111100001111000011110000111100001111,    len: 60
    n = (n | (n << 2))  & 0x3333333333333333 # binary: 11001100110011001100110011001100110011001100110011001100110011,  len: 62
    n = (n | (n << 1))  & 0x5555555555555555 # binary: 101010101010101010101010101010101010101010101010101010101010101, len: 63

    return n


def __part1by2_64(n):
    n = n & 0x1fffff                            # binary: 111111111111111111111,                                         len: 21
    n = (n | (n << 32)) & 0x1f00000000ffff   # binary: 11111000000000000000000000000000000001111111111111111,         len: 53
    n = (n | (n << 16)) & 0x1f0000ff0000ff   # binary: 11111000000000000000011111111000000000000000011111111,         len: 53
    n = (n | (n << 8))  & 0x100f00f00f00f00f # binary: 1000000001111000000001111000000001111000000001111000000001111, len: 61
    n = (n | (n << 4))  & 0x10c30c30c30c30c3 # binary: 1000011000011000011000011000011000011000011000011000011000011, len: 61
    n = (n | (n << 2))  & 0x1249249249249249 # binary: 1001001001001001001001001001001001001001001001001001001001001, len: 61

    return n


def __unpart1by1_64(n):
    n = n & 0x5555555555555555                  # binary: 101010101010101010101010101010101010101010101010101010101010101, len: 63
    n = (n ^ (n >> 1))  & 0x3333333333333333 # binary: 11001100110011001100110011001100110011001100110011001100110011,  len: 62
    n = (n ^ (n >> 2))  & 0x0f0f0f0f0f0f0f0f # binary: 111100001111000011110000111100001111000011110000111100001111,    len: 60
    n = (n ^ (n >> 4))  & 0x00ff00ff00ff00ff # binary: 11111111000000001111111100000000111111110000000011111111,        len: 56
    n = (n ^ (n >> 8))  & 0x0000ffff0000ffff # binary: 1111111111111111000000001111111111111111,                        len: 40
    n = (n ^ (n >> 16)) & 0x00000000ffffffff # binary: 11111111111111111111111111111111,                                len: 32
    return n


def __unpart1by2_64(n):
    n = n & 0x1249249249249249                  # binary: 1001001001001001001001001001001001001001001001001001001001001, len: 61
    n = (n ^ (n >> 2))  & 0x10c30c30c30c30c3 # binary: 1000011000011000011000011000011000011000011000011000011000011, len: 61
    n = (n ^ (n >> 4))  & 0x100f00f00f00f00f # binary: 1000000001111000000001111000000001111000000001111000000001111, len: 61
    n = (n ^ (n >> 8))  & 0x1f0000ff0000ff   # binary: 11111000000000000000011111111000000000000000011111111,         len: 53
    n = (n ^ (n >> 16)) & 0x1f00000000ffff   # binary: 11111000000000000000000000000000000001111111111111111,         len: 53
    n = (n ^ (n >> 32)) & 0x1fffff           # binary: 111111111111111111111,                                         len: 21
    return n


def interleave2d_32(data):
    return __part1by1_32(data[0]) | (__part1by1_32(data[1]) << 1)


def interleave2d_64(data):
    return __part1by1_64(data[0]) | (__part1by1_64(data[1]) << 1)


def interleave3d_32(data):
    return __part1by2_32(data[0]) | (__part1by2_32(data[1]) << 1) | (__part1by2_32(data[2]) << 2)


def deinterleave2d_32(n):
    return __unpart1by1_32(n), __unpart1by1_32(n >> 1)


def deinterleave2d_64(n):
    return __unpart1by1_64(n), __unpart1by1_64(n >> 1)


def deinterleave3d_32(n):
    if not isinstance(n, int):
        print('Usage: deinterleave2(n)')
        raise ValueError("Supplied arguments contain a non-integer!")

    return __unpart1by2_32(n), __unpart1by2_32(n >> 1), __unpart1by2_32(n >> 2)
