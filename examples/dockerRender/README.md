# dockerRender

The `trimesh` docker images are configured to use the [LLVMpipe](https://www.phoronix.com/scan.php?page=news_item&px=LLVMpipe-Mesa-19.0-Performance) software rasterizer and XVFB. Who knew LLVM also solved rasterization, somehow? The advantage of not using a GPU is so you can use it in "normal" cheap cloud instances (i.e. DigitalOcean). It also is WAY easier than configuring a GPU to be usable in a container (CUDA, EGL, etc). 

This example will render a PNG of a sphere, probably:
```
bash run.bash
```

It should print a bunch of log messages, and something like `rendered bytes: 49803`. If you `ctl-c` to exit the container, you should now see a PNG in the current directory:

```
mikedh@orbital:dockerRender$ ls
Dockerfile  README.md  output.png  render.py  render.supervisor.conf
```

It's a sphere!
