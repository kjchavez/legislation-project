import sys

# NOTE(kjchavez): We could also import * here, but we'd get so much stuff.
# == START MODELS ==
import baseline
import linear
# == END MODELS ==

def get_model_fn_by_name(name):
    submodule = getattr(sys.modules[__name__], name)
    return submodule.model_fn
