# docker_render

The `trimesh-x` docker images are configured to use the [LLVMpipe](https://www.phoronix.com/scan.php?page=news_item&px=LLVMpipe-Mesa-19.0-Performance) software rasterizer and XVFB. Who knew LLVM also solved rasterization, somehow? It's not as fast as a real GPU but it can do ~30FPS with a decent CPU which is probably plenty for analysis work.

The advantage of not using a GPU is so you can use it in "normal" cloud instances. Also whether `scene.save_image` is able to render anything ends up being very OS, GPU, and driver specific. With the docker images it should pretty much work on most clients.  Software rendering also is way easier than configuring a GPU to be usable in a container (CUDA, EGL, etc).


This will render a PNG of a sphere, probably:
```
bash run.bash
```

It should print a bunch of log messages, and something like `rendered bytes: 49803`. You should now see a PNG in the current directory:

```
mikedh@orbital:docker_render$ ls
Dockerfile  README.md  output.png  render.py  run.bash
```

It's a sphere!
