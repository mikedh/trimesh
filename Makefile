# See: https://clarkgrubb.com/makefile-style-guide
MAKEFLAGS += --warn-undefined-variables
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := all
.DELETE_ON_ERROR:
.SUFFIXES:

VERSION := $(shell python trimesh/version.py)
GIT_SHA := $(shell git rev-parse --short HEAD) 


IMAGE_NAME=trimesh/trimesh
DOCKER_REPO=docker.io

# the tags
TAG_LATEST=$(DOCKER_REPO)/$(IMAGE_NAME):latest
TAG_GIT_SHA=$(DOCKER_REPO)/$(IMAGE_NAME):$(GIT_SHA)
TAG_VERSION=$(DOCKER_REPO)/$(IMAGE_NAME):$(VERSION)

# This will output the help for each task
# See: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
.PHONY: help
help: ## Print usage help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
.DEFAULT_GOAL := help

# Use inline build cache when building with docker.
# See: https://github.com/moby/moby/issues/39003#issuecomment-852915070
.PHONY: build
build: ## Build the docker images
	DOCKER_BUILDKIT=1 \
	docker build \
		--target output \
		--tag $(TAG_LATEST) \
		--tag $(TAG_VERSION) \
		--tag $(TAG_GIT_SHA) \
		--cache-from $(TAG_LATEST) \
		--build-arg "BUILDKIT_INLINE_CACHE=1" \
		--build-arg "VERSION=$(VERSION)" \
		.

.PHONY: test
test: build ## Run unit tests inside Docker image
	docker run -v $(PWD):/tmp/trimesh -t $(TAG_LATEST) \
		pip install /tmp/trimesh[test] && \
		ls -altrsh /tmp && \
		pytest /tmp/trimesh/tests

.PHONY: bash
bash: build ## Start a bash terminal inside the image for debugging.
	docker run -it $(TAG_LATEST) /bin/bash

.PHONY: publish-docker
publish-docker: build ## Publish Docker images.
	docker push $(DOCKER_REPO)/$(IMAGE_NAME):latest
	docker push $(DOCKER_REPO)/$(IMAGE_NAME):$(VERSION)
