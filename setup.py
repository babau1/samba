import os, re
import shutil
from setuptools import setup, find_packages
from distutils.command.clean import clean as _clean
from distutils.command.sdist import sdist
import samba

# try:
#     import numpy
# except:
#     raise 'Cannot build  without numpy'
#     sys.exit()

# --------------------------------------------------------------------
# Clean target redefinition - force clean everything supprimer de la liste '^core\.*$',
relist = ['^.*~$', '^#.*#$', '^.*\.aux$', '^.*\.pyc$', '^.*\.o$']
reclean = []
USE_COPYRIGHT = True
try:
    from copyright import writeStamp, eraseStamp
except ImportError:
    USE_COPYRIGHT = False

###################
# Get Multimodal version
####################
def get_version():
    v_text = open('VERSION').read().strip()
    v_text_formted = '{"' + v_text.replace('\n', '","').replace(':', '":"')
    v_text_formted += '"}'
    v_dict = eval(v_text_formted)
    return v_dict["SamBA"]

########################
# Set Multimodal __version__
########################
def set_version(multiview_generator_dir, version):
    filename = os.path.join(multiview_generator_dir, '__init__.py')
    buf = ""
    for line in open(filename, "rb"):
        if not line.decode("utf8").startswith("__version__ ="):
            buf += line.decode("utf8")
    f = open(filename, "wb")
    f.write(buf.encode("utf8"))
    f.write(('__version__ = "%s"\n' % version).encode("utf8"))

for restring in relist:
    reclean.append(re.compile(restring))


def wselect(args, dirname, names):
    for n in names:
        for rev in reclean:
            if (rev.match(n)):
                os.remove("%s/%s" %(dirname, n))
        break


######################
# Custom clean command
######################
class clean(_clean):
    def walkAndClean(self):
        os.walk("..", wselect, [])
        pass

    def run(self):
        clean.run(self)
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('iw'):
            for filename in filenames:
                if (filename.endswith('.so') or
                        filename.endswith('.pyd') or
                        filename.endswith('.dll') or
                        filename.endswith('.pyc')):
                    os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))


##############################
# Custom sdist command
##############################
class m_sdist(sdist):
    """ Build source package

    WARNING : The stamping must be done on an default utf8 machine !
    """

    def run(self):
        if USE_COPYRIGHT:
            writeStamp()
            sdist.run(self)
            # eraseStamp()
        else:
            sdist.run(self)

def setup_package():
    """Setup function"""

    name = 'samba'
    version = get_version()
    dir = 'samba'
    set_version(dir, version)
    description = 'SamBA - Sample Boosting Classifier'
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.md'), encoding='utf-8') as readme:
        long_description = readme.read()
    group = 'dev'
    url = 'https://github.com/babau1/samba'.format(group, name)
    project_urls = {
        'Source': url,
        'Tracker': '{}/issues'.format(url)}
    author = 'Baptiste BAUVIN'
    author_email = 'baptiste.bauvin.work@gmail.com'
    license = 'GNU GPLv3'
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License'
        ' v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS'],
    keywords = ('machine learning, supervised learning, classification, '
                'ensemble methods, boosting, local')
    packages = find_packages()
    install_requires = ['numpy', 'scikit-learn>=0.19', 'scipy', "plotly",
                        "h5py", 'pyyaml', 'tabulate', 'pandas', "six", ]
    python_requires = '>=3.5'
    include_package_data = True
    setup(name=name,
          version=version,
          description=description,
          long_description=long_description,
          url=url,
          project_urls=project_urls,
          author=author,
          author_email=author_email,
          license=license,
          classifiers=classifiers,
          keywords=keywords,
          packages=packages,
          install_requires=install_requires,
          python_requires=python_requires,
          include_package_data=include_package_data,
          )


if __name__ == "__main__":
    setup_package()
