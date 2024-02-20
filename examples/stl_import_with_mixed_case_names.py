import trimesh

# Load stl file with mixed case name
mesh = trimesh.load_mesh("models/two_objects_mixed_case_names.stl")
# Print some geometry info to test
for k, v in mesh.geometry.items():
    print("Sub-mesh Name:", k, "\nSub-mesh Min X value", min(v.vertices[:, 0]))
