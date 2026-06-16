# Makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXGEN     ?= sphinx-apidoc
NBCONVERT     ?= jupyter
PYTHON        ?= python
PIP           ?= uv pip
BUILDDIR      = built
TEMPLATESDIR  = templates
# where to put generated RST files
GENDIR        = generate
CONTENT       = content
STATICDIR     = static
# where the test-coverage report is published in the built site
COVERAGEDIR   = $(BUILDDIR)/html/coverage


example_notebooks := $(wildcard ../examples/*.ipynb)
example_notebooks := $(filter-out ../examples/save_image.ipynb, $(example_notebooks))
example_names = $(foreach path, $(example_notebooks), $(basename $(notdir $(path))))
example_rsts = $(foreach name, $(example_names), examples.$(name).rst)


html: conf.py $(CONTENT)/index.rst trimesh.rst README.rst $(example_rsts) examples.rst .deps
	cp -R "$(STATICDIR)" "$(GENDIR)/static"
	@$(SPHINXBUILD) -M html "$(GENDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	echo "trimesh.org" > "$(BUILDDIR)/html/CNAME"
	touch "$(BUILDDIR)/html/.nojekyll"
	cp   "$(STATICDIR)/favicon.ico" "$(BUILDDIR)/html/favicon.ico" || true
	# render the styled coverage report + badge into the built site at /coverage/
	# run coverage from the repo root so its config and `trimesh` source resolve
	if [ -f ../.coverage ]; then \
		( cd .. && coverage html --data-file=.coverage -d "docs/$(COVERAGEDIR)" ) && \
		$(PYTHON) coverage_badge.py \
			"$$( cd .. && coverage report --format=total --precision=1 --data-file=.coverage )" \
			"$(COVERAGEDIR)/badge.svg" ; \
	else echo "no ../.coverage data file — skipping coverage report" ; fi

.deps: requirements.txt
	$(PIP) install -r requirements.txt
	$(PIP) freeze > .deps
	mkdir -p $(GENDIR)
	mkdir -p $(BUILDDIR)
	cp conf.py $(CONTENT)/* $(GENDIR)

examples.%.rst : ../examples/%.ipynb .deps
	$(NBCONVERT) nbconvert --execute --to rst --output-dir $(GENDIR) $<

examples.rst: .deps
	$(PYTHON) examples.py --source=../examples --target=$(GENDIR)/examples.rst

trimesh.rst: .deps
	$(SPHINXGEN) -eTf -t "$(TEMPLATESDIR)" -o "$(GENDIR)" ../trimesh

README.rst: ../README.md .deps
	pandoc --from=gfm --to=rst --output="$(GENDIR)/README.rst" ../README.md

clean:
	rm -rvf "$(BUILDDIR)" "$(GENDIR)"  .deps
