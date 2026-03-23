// load a scene saved as base64 encoded GLTF/GLB
var camera, controls, scene, renderer, tracklight;

/* Fit the camera to an object's bounding sphere
 *
 *
 */
function autoFit(obj, camera, controls) {
  // oriented bounding box
  const boundingBox = new THREE.Box3().setFromObject(obj);
  // get the bounding sphere of the OBB
  const boundingSphere = new THREE.Sphere();
  boundingBox.getBoundingSphere((target = boundingSphere));

  // object size / display size
  const scale = 1.0;
  // convert to radians and scale
  const angularSize = ((camera.fov * Math.PI) / 180) * scale;

  const distanceToCamera = boundingSphere.radius / Math.tan(angularSize);
  const len = Math.sqrt(
    Math.pow(distanceToCamera, 2) +
      Math.pow(distanceToCamera, 2) +
      Math.pow(distanceToCamera, 2)
  );

  camera.position.set(len, len, len);

  controls.update();

  camera.lookAt(boundingSphere.center);
  controls.target.set(
    boundingSphere.center.x,
    boundingSphere.center.y,
    boundingSphere.center.z
  );
  camera.updateProjectionMatrix();
}

/* Center the controls on the scene's bounding sphere
 *
 *
 */
function centerControls(obj, camera, controls) {
  // center control rotation on scene bounding box
  // oriented bounding box
  const boundingBox = new THREE.Box3().setFromObject(obj);
  // get the bounding sphere of the OBB
  const boundingSphere = new THREE.Sphere();
  boundingBox.getBoundingSphere((target = boundingSphere));

  // make sure we're orbiting around center of bounding sphere
  controls.update();
  controls.target.set(
    boundingSphere.center.x,
    boundingSphere.center.y,
    boundingSphere.center.z
  );
}

function init() {
  scene = new THREE.Scene();
  // white background
  scene.background = new THREE.Color(0xffffff);

  // add a light that will track the camera
  tracklight = new THREE.DirectionalLight(0xffffff, 1.75);
  scene.add(tracklight);

  // base64 encoded GLTF (GLB) scene
  base64_data =
    "Z2xURgIAAAAgBgAAFAUAAEpTT057InNjZW5lIjogMCwgInNjZW5lcyI6IFt7Im5vZGVzIjogWzBdfV0sICJhc3NldCI6IHsidmVyc2lvbiI6ICIyLjAiLCAiZ2VuZXJhdG9yIjogImdpdGh1Yi5jb20vbWlrZWRoL3RyaW1lc2gifSwgImFjY2Vzc29ycyI6IFt7ImJ1ZmZlclZpZXciOiAwLCAiY29tcG9uZW50VHlwZSI6IDUxMjUsICJjb3VudCI6IDM2LCAibWF4IjogWzddLCAibWluIjogWzBdLCAidHlwZSI6ICJTQ0FMQVIifSwgeyJidWZmZXJWaWV3IjogMSwgImNvbXBvbmVudFR5cGUiOiA1MTI2LCAiY291bnQiOiA4LCAidHlwZSI6ICJWRUMzIiwgImJ5dGVPZmZzZXQiOiAwLCAibWF4IjogWzAuNSwgMC41LCAwLjVdLCAibWluIjogWy0wLjUsIC0wLjUsIC0wLjVdfV0sICJtZXNoZXMiOiBbeyJuYW1lIjogImdlb21ldHJ5XzAiLCAicHJpbWl0aXZlcyI6IFt7ImF0dHJpYnV0ZXMiOiB7IlBPU0lUSU9OIjogMX0sICJpbmRpY2VzIjogMCwgIm1vZGUiOiA0LCAibWF0ZXJpYWwiOiAwfV19XSwgIm1hdGVyaWFscyI6IFt7InBick1ldGFsbGljUm91Z2huZXNzIjogeyJiYXNlQ29sb3JGYWN0b3IiOiBbMC40MDAwMDAwMDU5NjA0NjQ1LCAwLjQwMDAwMDAwNTk2MDQ2NDUsIDAuNDAwMDAwMDA1OTYwNDY0NSwgMS4wXSwgIm1ldGFsbGljRmFjdG9yIjogMC4wLCAicm91Z2huZXNzRmFjdG9yIjogMC4wfX1dLCAiY2FtZXJhcyI6IFt7Im5hbWUiOiAiY2FtZXJhX0dYUE9MRSIsICJ0eXBlIjogInBlcnNwZWN0aXZlIiwgInBlcnNwZWN0aXZlIjogeyJhc3BlY3RSYXRpbyI6IDEuMzMzMzMzMzMzMzMzMzMzMywgInlmb3YiOiAwLjc4NTM5ODE2MzM5NzQ0ODN9fV0sICJub2RlcyI6IFt7Im5hbWUiOiAid29ybGQiLCAiY2hpbGRyZW4iOiBbMSwgMl19LCB7Im1hdHJpeCI6IFsxLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMS4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDEuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAxLjBdLCAibmFtZSI6ICJnZW9tZXRyeV8wX0hCTjRaSUJKSEhYQSIsICJtZXNoIjogMH0sIHsibWF0cml4IjogWzEuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAxLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMS4wLCAwLjAsIDAuMCwgMC4wLCAxLjcwNzEwNjc4MTE4NjU0NzUsIDEuMF0sICJuYW1lIjogImNhbWVyYV9HWFBPTEUiLCAiY2FtZXJhIjogMH1dLCAiYnVmZmVycyI6IFt7ImJ5dGVMZW5ndGgiOiAyNDB9XSwgImJ1ZmZlclZpZXdzIjogW3siYnVmZmVyIjogMCwgImJ5dGVPZmZzZXQiOiAwLCAiYnl0ZUxlbmd0aCI6IDE0NH0sIHsiYnVmZmVyIjogMCwgImJ5dGVPZmZzZXQiOiAxNDQsICJieXRlTGVuZ3RoIjogOTZ9XX0gICAg8AAAAEJJTgABAAAAAwAAAAAAAAAEAAAAAQAAAAAAAAAAAAAAAwAAAAIAAAACAAAABAAAAAAAAAABAAAABwAAAAMAAAAFAAAAAQAAAAQAAAAFAAAABwAAAAEAAAADAAAABwAAAAIAAAAGAAAABAAAAAIAAAACAAAABwAAAAYAAAAGAAAABQAAAAQAAAAHAAAABQAAAAYAAAAAAAC/AAAAvwAAAL8AAAC/AAAAvwAAAD8AAAC/AAAAPwAAAL8AAAC/AAAAPwAAAD8AAAA/AAAAvwAAAL8AAAA/AAAAvwAAAD8AAAA/AAAAPwAAAL8AAAA/AAAAPwAAAD8=";

  // renderer
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // load GLTF data from base64 string
  loader = new THREE.GLTFLoader();
  loader.load(
    "data:text/plain;base64," + base64_data,
    // function will be called asynchronously after data is loaded
    function (gltf) {
      // add GLTF data to scene
      scene.add(gltf.scene);

      camera = gltf.cameras[0];

      // create trackball controls
      controls = new THREE.TrackballControls(camera, renderer.domElement);
      controls.rotateSpeed = 1.0;
      controls.zoomSpeed = 1.2;
      controls.panSpeed = 0.8;
      controls.noZoom = false;
      controls.noPan = false;
      controls.staticMoving = true;
      controls.dynamicDampingFactor = 0.3;
      controls.keys = [65, 83, 68];
      controls.addEventListener("change", render);

      // after loading make sure we're orbiting centroid
      centerControls(scene, camera, controls);

      // call initial render
      render();

      // add resize callbacks
      window.addEventListener("resize", onWindowResize, false);

      // enable controls
      animate();
      onWindowResize();
    }
  );
}

function onWindowResize() {
  // Handle window resizing
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  controls.handleResize();
  render();
}

function animate() {
  // Handle trackball controls
  requestAnimationFrame(animate);
  controls.update();
}

function render() {
  // we always want things lit for our preview window
  tracklight.position.copy(camera.position);
  renderer.render(scene, camera);
}

init();
