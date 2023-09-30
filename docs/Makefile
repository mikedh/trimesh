# Makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXGEN     ?= sphinx-apidoc
NBCONVERT     ?= jupyter
PYTHON        ?= python
PIP           ?= pip
SOURCEDIR     = .
BUILDDIR      = built
TEMPLATESDIR  = templates
# where to put generated RST files
GENDIR        = generated
STATICDIR     = _static


example_notebooks := $(wildcard ../examples/*.ipynb)
example_notebooks := $(filter-out ../examples/save_image.ipynb, $(example_notebooks))

example_names = $(foreach path, $(example_notebooks), $(basename $(notdir $(path))))

example_rsts = $(foreach name, $(example_names), examples.$(name).rst)

html: conf.py index.rst trimesh.rst README.rst examples.rst $(example_rsts) .deps
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	touch "$(BUILDDIR)/html/.nojekyll"
	echo "trimesh.org" > "$(BUILDDIR)/html/CNAME"
	mv "$(BUILDDIR)/html/_static/images" "$(BUILDDIR)/html/images" || true
	cp "$(STATICDIR)/favicon.ico" "$(BUILDDIR)/html/favicon.ico" || true

.deps: requirements.txt
	$(PIP) install -r requirements.txt
	$(PIP) freeze > .deps
	mkdir -p $(GENDIR)
	mkdir -p $(BUILDDIR)

examples.%.rst : ../examples/%.ipynb .deps
	$(NBCONVERT) nbconvert --execute --to rst --output-dir $(GENDIR) $<

examples.rst: .deps
	$(PYTHON) "examples.py" "--source=../examples" "--target=$(GENDIR)/examples.rst"

trimesh.rst: .deps
	$(SPHINXGEN) -eTf -t "$(TEMPLATESDIR)" -o "$(GENDIR)" ../trimesh

README.rst: ../README.md .deps
	pandoc --from=gfm --to=rst --output="$(GENDIR)/README.rst" ../README.md

clean:
	rm -rvf "$(BUILDDIR)" "$(GENDIR)" .deps
