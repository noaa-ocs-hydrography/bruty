# -*- coding: utf-8 -*-
#
# Pydro documentation build configuration file, created by
# sphinx-quickstart on Tue Nov 28 16:43:03 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

import os
import pathlib
import sys
import types

import re
import sphinx
import mock

import sphinx_rtd_theme

# prevents these modules from being loaded when creating docs - so RTD doesn't have to install everything for temporary environment
MOCK_MODULES = ['scipy', 'matplotlib', 'matplotlib.pyplot', 'scipy.interpolate', 'osgeo', 'gdal', 'osr', 'HSTB.drivers.bag', 'bag']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

top_classes = "collections.abc.MutableSequence, bruty.abstract.ABC, bruty.abstract.VABC"

p, f = os.path.split(__file__)
root_p = os.path.normpath(p)
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("..\\.."))

# rst_prolog = """
# .. |DOCS_DIR| replace:: %s
# .. include:: %s\\_Globals\\substitutions.txt
# """ % (root_p.replace("\\", "/") + "/", root_p)

# rst_epilog = """
# .. include:: %s\\_Globals\\footer.rst
# """ % root_p

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = '%s/_static/pydro_sidebar.png' % root_p

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# html_favicon = '%s/_static/pydro.ico' % root_p


# watch out, used ratio (without reading spec) and it made the image go to two pages which fails with sphinx
inheritance_graph_attrs = dict(rankdir="TB", size='"24.0 36.0"')  # make the pictures bigger, default is "8.0 12.0"

def set_top_classes(*args, **kys):
    """The automodapi is making .rst files for each class but the inheritance graph is
    going back too far, so stopping it by inserting :top-class: into each auto-generated .rst
    This could probably be done in the doctree object but it wasn't immediately apparent where that info is stored.
    Using the doctree-read event is too late, so we revise the api directory during the builder-inited event.
    """
    # path = pathlib.Path(doctree['source'])
    # if path.parent.name == 'api':
    for path in pathlib.Path(root_p).joinpath('api').glob('*.rst'):
        rst = open(path, 'r').read()
        loc = rst.find('.. inheritance-diagram::')
        if loc > 0:
            print(loc, path.name)
            end_diagram = rst.find("\n\n", loc)
            if ':top-classes:' not in rst[loc:end_diagram].lower():
                out = open(path, 'w')
                out.write(rst[:end_diagram])
                out.write('\n      :top-classes: '+top_classes)
                out.write(rst[end_diagram:])


def setup(app):
    # app.connect("doctree-read", check)  # slip in top-classes
    app.connect("builder-inited", set_top_classes)  # slip in top-classes
    pass
    # app.add_stylesheet('pydro_custom.css')  # may also be an URL


html_show_sourcelink = False
html_copy_source = False

html_sidebars = {'**': ['localtoc.html', 'relations.html', 'searchbox.html'], }  # , 'download_page.html'


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# http://www.sphinx-doc.org/en/stable/ext/ifconfig.html

try:
    import sphinxcontrib.fulltoc
    full_toc = 'sphinxcontrib.fulltoc'
except:
    # Giuseppe and I installed in different places
    try:
        import sphinxcontrib_fulltoc.fulltoc
        full_toc = 'sphinxcontrib_fulltoc.fulltoc'
    except:
        full_toc = ""

extensions = ['sphinx.ext.imgmath',
              'sphinx.ext.githubpages',
              'sphinx.ext.extlinks',
              'sphinx.ext.ifconfig',
#              full_toc,
              'sphinx.ext.graphviz',
              'sphinx.ext.napoleon',
              # 'recommonmark',
              'm2r2',
              'sphinx_autodoc_typehints',
              'sphinx.ext.autodoc',
              "sphinx_rtd_theme",
              ]

extensions.extend([ # 'sphinx.ext.autosummary',
                   'sphinx_automodapi.automodapi',
              ])
automodsumm_inherited_members = True
automodapi_inheritance_diagram = False
numpydoc_show_class_members = False  # prevents duplication per automodapi docs
# autosummary_generate = True


bg_dot_path = r"c:\PydroTrunk\Miniconda36\envs\Pydro367\Library\bin\graphviz\dot.exe"
if os.path.exists(bg_dot_path):  # on rtd site use the default dot install
    graphviz_dot = bg_dot_path

autodoc_default_options = {
   # 'members': 'var1, var2',
   'member-order': 'bysource',
   'special-members': '__init__',
   'undoc-members': True,
   'exclude-members': '__weakref__',
   'top-classes': top_classes,
#    'inherited-members': False,
}

# autodoc_default_flags = [
#         # Make sure that any autodoc declarations show the right members
#         "members",
#         #"inherited-members",
#         "private-members",
#         "show-inheritance",
# ]

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
# source_suffix = '.rst'
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# adds figure numbers automatically
numfig = True
# numfig_secnum_depth = (2)
# numfig_format = {'figure': 'Figure %s', 'table': 'Table %s', 'code-block': 'Code Block %s'}

# General information about the project.


def set_project_names(prj, auth=u'Hydrographic Systems and Technology Branch', g={}):
    if not g:
        g = globals()
    g['project'] = prj
    g['author'] = auth
    # Output file base name for HTML help builder.
    g['htmlhelp_basename'] = prj + '_doc'
    # Grouping the document tree into LaTeX files. List of tuples
    # (source start file, target name, title,
    #  author, documentclass [howto, manual, or own class]).
    g['latex_documents'] = [
        (g['master_doc'], prj + '.tex', prj + u' Documentation',
         u'Hydrographic Systems and Technology Branch', 'manual'),
    ]
    # -- Options for manual page output ---------------------------------------
    # One entry per manual page. List of tuples
    # (source start file, name, description, authors, manual section).
    g['man_pages'] = [
        (g['master_doc'], prj, prj + u' Documentation',
         [g['author']], 1)
    ]

    # -- Options for Texinfo output -------------------------------------------
    # Grouping the document tree into Texinfo files. List of tuples
    # (source start file, target name, title, author,
    #  dir menu entry, description, category)
    g['texinfo_documents'] = [
        (g['master_doc'], prj, prj + u' Documentation',
         g['author'], prj, 'One line description of project.',
         'Miscellaneous'),
    ]


set_project_names(u'Bruty')
html_show_copyright = False
copyright = u'None - %s is not subject to Copyrights (though other distributed pieces may be)' % project

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = u''
# The full version, including alpha/beta/rc tags.
release = u''

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["sphinx_dir_template/*", '_build', '_build_offline', '_build_online', 'Thumbs.db', '.DS_Store']


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'classic'
html_theme = "sphinx_rtd_theme"
_style = str(pathlib.Path(__file__).parent.joinpath('templates/full_width_page_multiline_tables.css'))
# These folders are copied to the documentation's HTML output
html_static_path = ['templates']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'full_width_page_multiline_tables.css',
]
# print(html_style, os.path.exists(html_style))

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}
# html_theme_options = {
#     'sidebarbgcolor': "#054698",
#     'relbarbgcolor': "#29318B",
#     'footerbgcolor': "#29318B",
#     'linkcolor': "#0098DA",
#     'visitedlinkcolor': "#054698",
#     'sidebarlinkcolor': "#E0F2FF",
#     'body_max_width':'none',
#     # 'relbarlinkcolor' : "#0098DA",
#     # bgcolor
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


# -- Options for HTMLHelp output ------------------------------------------


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}
