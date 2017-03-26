import argparse
import os
import lxml.etree as ET

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uscode", '-c',
            default=os.path.join(os.environ['HOME'],"data/uslaw"),
            help="Directory holding U.S. code in XML format")
    parser.add_argument("--out", '-o', default='uscode_text',
            help="Output directory for text format U.S. code")
    parser.add_argument("--xslt", '-t', default="title.xsl")

    return parser.parse_args()

args = parse_args()
titles = [os.path.join(args.uscode, t) for t in os.listdir(args.uscode)]
print "Found %d titles" % len(titles)

xslt = ET.parse(args.xslt)
transform = ET.XSLT(xslt)

if not os.path.isdir(args.out):
    os.makedirs(args.out)

def with_txt_ext(filename):
    return filename.rsplit('.', 1)[0] + '.txt'

def transform_title(title):
    filename = os.path.basename(title)
    print "Transforming %s to text format." % filename
    outfile = os.path.join(args.out, with_txt_ext(filename))
    dom = ET.parse(title)
    text_format = transform(dom)
    with open(outfile, 'w') as fp:
        fp.write(str(text_format))

for title in titles:
    transform_title(title)
