"""ASCII, Dammit

Stupid library to turn MS chars (like smart quotes) and ISO-Latin
chars into ASCII, dammit. Will do plain text approximations, or more
accurate HTML representations. Can also be jiggered to just fix the
smart quotes and leave the rest of ISO-Latin alone.

Sources:
 http://www.cs.tut.fi/~jkorpela/latin1/all.html
 http://www.webreference.com/html/reference/character/isolat1.html

1.0 Initial Release (2004-11-28)

The author hereby irrevocably places this work in the public domain.
To the extent that this statement does not divest the copyright,
the copyright holder hereby grants irrevocably to every recipient
all rights in this work otherwise reserved under copyright.
"""

__author__ = "Leonard Richardson (leonardr@segfault.org)"
__version__ = "$Revision: 1.3 $"
__date__ = "$Date: 2009/04/28 10:45:03 $"
__license__ = "Public domain"

import re
import string
import types

CHARS = { '\x80' : ('EUR', 'euro'),
          '\x81' : ' ',
          '\x82' : (',', 'sbquo'),
          '\x83' : ('f', 'fnof'),
          '\x84' : (',,', 'bdquo'),
          '\x85' : ('...', 'hellip'),
          '\x86' : ('+', 'dagger'),
          '\x87' : ('++', 'Dagger'),
          '\x88' : ('^', 'caret'),
          '\x89' : '%',
          '\x8A' : ('S', 'Scaron'),
          '\x8B' : ('<', 'lt;'),
          '\x8C' : ('OE', 'OElig'),
          '\x8D' : '?',
          '\x8E' : 'Z',
          '\x8F' : '?',
          '\x90' : '?',
          '\x91' : ("'", 'lsquo'),
          '\x92' : ("'", 'rsquo'),
          '\x93' : ('"', 'ldquo'),
          '\x94' : ('"', 'rdquo'),
          '\x95' : ('*', 'bull'),
          '\x96' : ('-', 'ndash'),
          '\x97' : ('--', 'mdash'),
          '\x98' : ('~', 'tilde'),
          '\x99' : ('(TM)', 'trade'),
          '\x9a' : ('s', 'scaron'),
          '\x9b' : ('>', 'gt'),
          '\x9c' : ('oe', 'oelig'),
          '\x9d' : '?',
          '\x9e' : 'z',
          '\x9f' : ('Y', 'Yuml'),
          '\xa0' : (' ', 'nbsp'),
          '\xa1' : ('!', 'iexcl'),
          '\xa2' : ('c', 'cent'),
          '\xa3' : ('GBP', 'pound'),
          '\xa4' : ('$', 'curren'), #This approximation is especially lame.
          '\xa5' : ('YEN', 'yen'),
          '\xa6' : ('|', 'brvbar'),
          '\xa7' : ('S', 'sect'),
          '\xa8' : ('..', 'uml'),
          '\xa9' : ('', 'copy'),
          '\xaa' : ('(th)', 'ordf'),
          '\xab' : ('<<', 'laquo'),
          '\xac' : ('!', 'not'),
          '\xad' : (' ', 'shy'),
          '\xae' : ('(R)', 'reg'),
          '\xaf' : ('-', 'macr'),
          '\xb0' : ('o', 'deg'),
          '\xb1' : ('+-', 'plusmm'),
          '\xb2' : ('2', 'sup2'),
          '\xb3' : ('3', 'sup3'),
          '\xb4' : ("'", 'acute'),
          '\xb5' : ('u', 'micro'),
          '\xb6' : ('P', 'para'),
          '\xb7' : ('*', 'middot'),
          '\xb8' : (',', 'cedil'),
          '\xb9' : ('1', 'sup1'),
          '\xba' : ('(th)', 'ordm'),
          '\xbb' : ('>>', 'raquo'),
          '\xbc' : ('1/4', 'frac14'),
          '\xbd' : ('1/2', 'frac12'),
          '\xbe' : ('3/4', 'frac34'),
          '\xbf' : ('?', 'iquest'),          
          '\xc0' : ('A', "Agrave"),
          '\xc1' : ('A', "Aacute"),
          '\xc2' : ('A', "Acirc"),
          '\xc3' : ('A', "Atilde"),
          '\xc4' : ('A', "Auml"),
          '\xc5' : ('A', "Aring"),
          '\xc6' : ('AE', "Aelig"),
          '\xc7' : ('C', "Ccedil"),
          '\xc8' : ('E', "Egrave"),
          '\xc9' : ('E', "Eacute"),
          '\xca' : ('E', "Ecirc"),
          '\xcb' : ('E', "Euml"),
          '\xcc' : ('I', "Igrave"),
          '\xcd' : ('I', "Iacute"),
          '\xce' : ('I', "Icirc"),
          '\xcf' : ('I', "Iuml"),
          '\xd0' : ('D', "Eth"),
          '\xd1' : ('N', "Ntilde"),
          '\xd2' : ('O', "Ograve"),
          '\xd3' : ('O', "Oacute"),
          '\xd4' : ('O', "Ocirc"),
          '\xd5' : ('O', "Otilde"),
          '\xd6' : ('O', "Ouml"),
          '\xd7' : ('*', "times"),
          '\xd8' : ('O', "Oslash"),
          '\xd9' : ('U', "Ugrave"),
          '\xda' : ('U', "Uacute"),
          '\xdb' : ('U', "Ucirc"),
          '\xdc' : ('U', "Uuml"),
          '\xdd' : ('Y', "Yacute"),
          '\xde' : ('b', "Thorn"),
          '\xdf' : ('B', "szlig"),
          '\xe0' : ('a', "agrave"),
          '\xe1' : ('a', "aacute"),
          '\xe2' : ('a', "acirc"),
          '\xe3' : ('a', "atilde"),
          '\xe4' : ('a', "auml"),
          '\xe5' : ('a', "aring"),
          '\xe6' : ('ae', "aelig"),
          '\xe7' : ('c', "ccedil"),
          '\xe8' : ('e', "egrave"),
          '\xe9' : ('e', "eacute"),
          '\xea' : ('e', "ecirc"),
          '\xeb' : ('e', "euml"),
          '\xec' : ('i', "igrave"),
          '\xed' : ('i', "iacute"),
          '\xee' : ('i', "icirc"),
          '\xef' : ('i', "iuml"),
          '\xf0' : ('o', "eth"),
          '\xf1' : ('n', "ntilde"),
          '\xf2' : ('o', "ograve"),
          '\xf3' : ('o', "oacute"),
          '\xf4' : ('o', "ocirc"),
          '\xf5' : ('o', "otilde"),
          '\xf6' : ('o', "ouml"),
          '\xf7' : ('/', "divide"),
          '\xf8' : ('o', "oslash"),
          '\xf9' : ('u', "ugrave"),
          '\xfa' : ('u', "uacute"),
          '\xfb' : ('u', "ucirc"),
          '\xfc' : ('u', "uuml"),
          '\xfd' : ('y', "yacute"),
          '\xfe' : ('b', "thorn"),
          '\xff' : ('y', "yuml"),
          }

def _makeRE(limit):
    """Returns a regular expression object that will match special characters
    up to the given limit."""
    return re.compile("([\x80-\\x%s])" % limit, re.M)
ALL = _makeRE('ff')
ONLY_WINDOWS = _makeRE('9f')

def _replHTML(match):
    "Replace the matched character with its HTML equivalent."
    return _repl(match, 1)
          
def _repl(match, html=0):
    "Replace the matched character with its HTML or ASCII equivalent."
    g = match.group(0)
    a = CHARS.get(g,g)
    if type(a) == types.TupleType:
        a = a[html]
        if html:
            a = '&' + a + ';'
    return a

def _dammit(t, html=0, fixWindowsOnly=0):
    "Turns ISO-Latin-1 into an ASCII representation, dammit."

    r = ALL
    if fixWindowsOnly:
        r = ONLY_WINDOWS
    m = _repl
    if html:
        m = _replHTML

    return re.sub(r, m, t)

def asciiDammit(t, fixWindowsOnly=0):
    "Turns ISO-Latin-1 into a plain ASCII approximation, dammit."
    return _dammit(t, 0, fixWindowsOnly)

def htmlDammit(t, fixWindowsOnly=0):
    "Turns ISO-Latin-1 into plain ASCII with HTML codes, dammit."
    return _dammit(t, 1, fixWindowsOnly=fixWindowsOnly)

def demoronise(t):
    """Helper method named in honor of the original smart quotes
    remover, The Demoroniser:

    http://www.fourmilab.ch/webtools/demoroniser/"""
    return asciiDammit(t, 1)

if __name__ == '__main__':

    french = '\x93Sacr\xe9 bleu!\x93'
    print "First we mangle some French."
    print asciiDammit(french)
    print htmlDammit(french)

    print
    print "And now we fix the MS-quotes but leave the French alone."
    print demoronise(french)
    print htmlDammit(french, 1)
