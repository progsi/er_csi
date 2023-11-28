import unicodedata


class UnicodeNormalize(object):
    """Class to normalize to basic latin characters. This is useful in the case
    of YouTube metadata, since sometimes utf encoded fonts are used.
    For instance, ones that are generatable: https://fancy-fonts.com/youtube-fonts/
    Args:
        object (_type_): _description_
    """
    def __call__(self, text):
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
