"""A reply's address: arithmetic coding with the model's own odds, written in Babel's alphabet.

Every possible reply has an address. Under a weighting (a "measure": the model's odds at each step, with or
without the stream), a likely reply has a short address and an unlikely one a long one; its length in bits is the
reply's price. Decoding the address with the same measure gives back exactly the same reply. Decoding it with a
different measure (the model alone, or the stream on a later day) reads the same number as a different text:
the address only means something under the weighting it was made with.

The address is written in Babel's 29 characters (base 29), so it is itself a short text from the Library. In Babel
proper (no measure) a reply's address is the reply, character for character.

All arithmetic is exact (Python integers). Odds are quantised to integer counts the same way on both sides, so
encode and decode agree bit for bit as long as the measure returns the same floats.
"""
import numpy as np

ALPHABET = "abcdefghijklmnopqrstuvwxyz ,."           # Babel's 29, as in physical/cipher_wheel.py
BASE = len(ALPHABET)
TOTAL = 2 ** 32                                       # resolution of the quantised odds at each step


def quantise(p):
    """Integer counts for every token: at least 1 each (so any text stays reachable), the rest by probability."""
    p = np.asarray(p, dtype=np.float64)
    freq = np.floor(p / p.sum() * (TOTAL - len(p))).astype(np.int64) + 1
    cum = np.zeros(len(freq) + 1, dtype=np.int64)
    np.cumsum(freq, out=cum[1:])
    return freq, cum


def encode(measure, prefix, tokens):
    """The shortest base-29 fraction inside the reply's interval. measure(ids) -> odds over the vocabulary."""
    L, W, S = 0, 1, 1                                 # the interval [L/S, (L+W)/S)
    ids = list(prefix)
    for t in tokens:
        freq, cum = quantise(measure(ids))
        T = int(cum[-1])
        L, W, S = L * T + W * int(cum[t]), W * int(freq[t]), S * T
        ids.append(int(t))
    k = 1
    while True:
        X = -(-L * BASE ** k // S)                    # ceil(L / S · 29^k)
        if X * S < (L + W) * BASE ** k:
            return X, k
        k += 1


def decode(measure, prefix, X, k, n):
    """Read n tokens at the address X / 29^k under a measure."""
    L, W, S = 0, 1, 1
    D = BASE ** k
    ids, out = list(prefix), []
    for _ in range(n):
        freq, cum = quantise(measure(ids))
        T = int(cum[-1])
        val = (X * S * T - L * T * D) // (W * D)      # where the address falls among this step's counts
        j = int(np.searchsorted(cum, val, side="right")) - 1
        L, W, S = L * T + W * int(cum[j]), W * int(freq[j]), S * T
        out.append(j); ids.append(j)
    return out


def to_text(X, k):
    """The address as k characters of Babel's alphabet."""
    digits = []
    for _ in range(k):
        X, r = divmod(X, BASE)
        digits.append(ALPHABET[r])
    return "".join(reversed(digits))


def from_text(s):
    X = 0
    for c in s:
        X = X * BASE + ALPHABET.index(c)
    return X, len(s)
