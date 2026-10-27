// The web GLB of the Amina statue: the baked mesh (tools/webmesh.py), its grey texture (tools/bake.py) as WebP,
// unlit (KHR_materials_unlit: the texture is the photograph, light and all), meshopt-compressed.
//
//   node pack.mjs <dir with positions.f32, uvs.f32, faces.u32> <texture.webp> <out.glb>
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { Document, NodeIO } from "@gltf-transform/core";
import { EXTMeshoptCompression, EXTTextureWebP, KHRMaterialsUnlit, KHRMeshQuantization } from "@gltf-transform/extensions";
import { meshopt, reorder, quantize } from "@gltf-transform/functions";
import { MeshoptEncoder, MeshoptDecoder } from "meshoptimizer";

const [dir, tex, out] = process.argv.slice(2);
const f32 = (name) => new Float32Array(readFileSync(join(dir, name)).buffer.slice(0));
const positions = f32("positions.f32");
const uvs = f32("uvs.f32");
const faces = new Uint32Array(readFileSync(join(dir, "faces.u32")).buffer.slice(0));

const doc = new Document();
const buffer = doc.createBuffer();
const unlit = doc.createExtension(KHRMaterialsUnlit);
doc.createExtension(EXTTextureWebP).setRequired(true);
const texture = doc.createTexture("amina").setImage(new Uint8Array(readFileSync(tex))).setMimeType("image/webp");
const material = doc.createMaterial("amina").setBaseColorTexture(texture).setExtension("KHR_materials_unlit", unlit.createUnlit()).setDoubleSided(true);
const prim = doc
  .createPrimitive()
  .setAttribute("POSITION", doc.createAccessor().setType("VEC3").setArray(positions).setBuffer(buffer))
  .setAttribute("TEXCOORD_0", doc.createAccessor().setType("VEC2").setArray(uvs).setBuffer(buffer))
  .setIndices(doc.createAccessor().setType("SCALAR").setArray(faces).setBuffer(buffer))
  .setMaterial(material);
const mesh = doc.createMesh("amina").addPrimitive(prim);
doc.createScene("amina").addChild(doc.createNode("amina").setMesh(mesh));

await MeshoptEncoder.ready;
doc.createExtension(KHRMeshQuantization);
await doc.transform(
  reorder({ encoder: MeshoptEncoder }),
  // 14-bit positions over a ~1-unit statue: ~0.06 mm a step at life size, far under a print dot
  quantize({ quantizePosition: 14, quantizeTexcoord: 14 }),
  meshopt({ encoder: MeshoptEncoder, level: "high" }),
);
const io = new NodeIO().registerExtensions([EXTMeshoptCompression, EXTTextureWebP, KHRMaterialsUnlit, KHRMeshQuantization]).registerDependencies({ "meshopt.decoder": MeshoptDecoder, "meshopt.encoder": MeshoptEncoder });
await io.write(out, doc);
console.log("wrote", out, `${positions.length / 3} vertices, ${faces.length / 3} faces`);
