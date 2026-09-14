import * as THREE from "three";

type AtlasManifest = {
  count: number;
  brainCount: number;
  dataset: string;
};

export type BrainActivity = {
  signal: "emitted_simulated_spike_count";
  values: [number, string, number, number][];
};

export type MappedNeuron = {
  neuron_index: number;
  body_id: string;
  visible_atlas_row: number | null;
  visible_soma: boolean;
};

export type Neighborhood = {
  nodes: {neuron_index: number; body_id: string; atlas_row: number; seed: boolean}[];
  edges: {
    source_neuron_index: number;
    target_neuron_index: number;
    synapse_count: number;
  }[];
};

export type BrainView = {
  updateActivity(frame: BrainActivity | null): void;
  setSelection(inputs: MappedNeuron[], outputs: MappedNeuron[], watch: MappedNeuron[]): void;
  setNeighborhood(value: Neighborhood | null): void;
  resetView(): void;
  setOrbit(enabled: boolean): void;
  setRendering(enabled: boolean): void;
  getPerformance(): {render_count: number; mean_cpu_submit_ms: number; max_cpu_submit_ms: number};
  dispose(): void;
};

type LoadedAtlas = {
  manifest: AtlasManifest;
  positions: Float32Array;
  ids: Uint32Array;
  groups: Uint8Array;
};

const BASE = "/assets/brain-atlas/";

async function loadAtlas(signal: AbortSignal): Promise<LoadedAtlas> {
  const read = async (name: string): Promise<Response> => {
    const response = await fetch(`${BASE}${name}`, {signal});
    if (!response.ok) throw Error(`Atlas could not load: ${name}`);
    return response;
  };
  const manifest = await (await read("manifest.json")).json() as AtlasManifest;
  const buffers = await Promise.all(
    ["positions.bin", "ids.bin", "groups.bin"].map(async (name) =>
      (await read(name)).arrayBuffer()),
  );
  const positions = new Float32Array(buffers[0]);
  const ids = new Uint32Array(buffers[1]);
  const groups = new Uint8Array(buffers[2]);
  if (manifest.dataset !== "male-cns:v1.0" ||
      positions.length !== manifest.count * 3 || ids.length !== manifest.count ||
      groups.length !== manifest.count) {
    throw Error("Atlas identity or file lengths do not match the manifest.");
  }
  let visible = 0;
  for (const group of groups) if (group < 3) visible++;
  if (visible !== manifest.brainCount) throw Error("Atlas brain selection is invalid.");
  return {manifest, positions, ids, groups};
}

/**
 * Measured soma renderer adapted from fly-connectome-template BrainScene.
 * Activity is joined by exact MaleCNS body ID; spatial proximity is never used.
 */
export function createBrainView(host: HTMLElement, stateLabel: HTMLElement): BrainView {
  const abort = new AbortController();
  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-3, 3, 2, -2, 0.01, 100);
  const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  host.appendChild(renderer.domElement);
  const anatomy = new THREE.Group();
  scene.add(anatomy);

  let geometry: THREE.BufferGeometry | null = null;
  let material: THREE.ShaderMaterial | null = null;
  let lines: THREE.LineSegments | null = null;
  let size = new THREE.Vector3(5, 2, 1);
  let pointByBodyId = new Map<string, number>();
  let pointByNeuronIndex = new Map<number, number>();
  let pointByAtlasRow = new Int32Array(0);
  let transformedPositions = new Float32Array(0);
  let activity = new Float32Array(0);
  let selection = new Float32Array(0);
  let activePoints: number[] = [];
  let orbit = true;
  let rendering = true;
  let disposed = false;
  let renderCount = 0;
  let renderCpuMs = 0;
  let maximumRenderCpuMs = 0;
  let held = false;
  let lastX = 0;
  let lastY = 0;

  const renderScene = () => {
    const started = performance.now();
    renderer.render(scene, camera);
    const elapsed = performance.now() - started;
    renderCount++;
    renderCpuMs += elapsed;
    maximumRenderCpuMs = Math.max(maximumRenderCpuMs, elapsed);
  };

  const fit = () => {
    const {width, height} = host.getBoundingClientRect();
    renderer.setSize(Math.max(1, width), Math.max(1, height), false);
    const aspect = Math.max(1, width) / Math.max(1, height);
    const yawRadius = Math.hypot(size.x, size.z) / 2;
    const tiltedHeight = Math.abs(Math.cos(anatomy.rotation.x)) * size.y / 2 +
      Math.abs(Math.sin(anatomy.rotation.x)) * yawRadius;
    const halfHeight = Math.max(tiltedHeight, yawRadius / aspect) * 1.08;
    camera.top = halfHeight;
    camera.bottom = -halfHeight;
    camera.left = -halfHeight * aspect;
    camera.right = halfHeight * aspect;
    camera.position.set(0, 0, 10);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
    if (rendering) renderScene();
  };

  const repaint = () => {
    if (!disposed && rendering) renderScene();
  };

  void loadAtlas(abort.signal).then(({manifest, positions, ids, groups}) => {
    if (disposed) return;
    const xyz: number[] = [];
    const bounds = new THREE.Box3();
    pointByAtlasRow = new Int32Array(ids.length);
    pointByAtlasRow.fill(-1);
    for (let atlasRow = 0; atlasRow < ids.length; atlasRow++) {
      if (groups[atlasRow] >= 3) continue;
      const point = new THREE.Vector3(
        positions[atlasRow * 3],
        -positions[atlasRow * 3 + 1],
        -positions[atlasRow * 3 + 2],
      );
      const pointIndex = xyz.length / 3;
      pointByAtlasRow[atlasRow] = pointIndex;
      pointByBodyId.set(String(ids[atlasRow]), pointIndex);
      xyz.push(point.x, point.y, point.z);
      bounds.expandByPoint(point);
    }
    const center = bounds.getCenter(new THREE.Vector3());
    size = bounds.getSize(new THREE.Vector3());
    const scale = 5 / Math.max(size.x, size.y, size.z);
    for (let index = 0; index < xyz.length; index += 3) {
      xyz[index] = (xyz[index] - center.x) * scale;
      xyz[index + 1] = (xyz[index + 1] - center.y) * scale;
      xyz[index + 2] = (xyz[index + 2] - center.z) * scale;
    }
    size.multiplyScalar(scale);
    transformedPositions = new Float32Array(xyz);
    activity = new Float32Array(manifest.brainCount);
    selection = new Float32Array(manifest.brainCount);
    geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(transformedPositions, 3));
    geometry.setAttribute("activity", new THREE.BufferAttribute(activity, 1));
    geometry.setAttribute("selection", new THREE.BufferAttribute(selection, 1));
    material = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      uniforms: {pixelRatio: {value: Math.min(window.devicePixelRatio, 1.5)}},
      vertexShader: `attribute float activity; attribute float selection;
        varying float strength; varying float selected; uniform float pixelRatio;
        void main() { strength=activity; selected=selection;
        gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);
        gl_PointSize=(0.9+strength*2.4+(selected>0.0?1.3:0.0))*pixelRatio; }`,
      fragmentShader: `varying float strength; varying float selected;
        void main() { float r=length(gl_PointCoord-vec2(.5)); if(r>.5) discard;
        vec3 base=vec3(.12,.35,.75);
        if(selected==1.0) base=vec3(.79,.96,.37);
        else if(selected==2.0) base=vec3(1.,.55,.27);
        else if(selected==3.0) base=vec3(.78,.61,1.);
        vec3 color=mix(base,vec3(.25,.95,1.),strength);
        color=mix(color,vec3(1.),smoothstep(.6,1.,strength));
        gl_FragColor=vec4(color,(.26+.7*max(strength,selected>0.0?.38:0.0))*(1.-smoothstep(.18,.5,r))); }`,
    });
    anatomy.add(new THREE.Points(geometry, material));
    stateLabel.textContent = "atlas ready";
    fit();
  }).catch((error: unknown) => {
    if (!disposed) stateLabel.textContent = error instanceof Error ? error.message : "atlas unavailable";
  });

  const observer = new ResizeObserver(fit);
  observer.observe(host);
  const down = (event: PointerEvent) => {
    held = true;
    lastX = event.clientX;
    lastY = event.clientY;
    renderer.domElement.setPointerCapture(event.pointerId);
  };
  const move = (event: PointerEvent) => {
    if (!held) return;
    anatomy.rotation.y += (event.clientX - lastX) * 0.006;
    anatomy.rotation.x += (event.clientY - lastY) * 0.006;
    lastX = event.clientX;
    lastY = event.clientY;
    repaint();
  };
  const up = () => { held = false; };
  renderer.domElement.addEventListener("pointerdown", down);
  renderer.domElement.addEventListener("pointermove", move);
  renderer.domElement.addEventListener("pointerup", up);
  renderer.domElement.addEventListener("pointercancel", up);

  let animationFrame = 0;
  let previous = performance.now();
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const animate = (now: number) => {
    const elapsed = Math.min(0.05, (now - previous) / 1000);
    previous = now;
    if (rendering && !document.hidden) {
      if (orbit && !held && !reducedMotion.matches) anatomy.rotation.y += elapsed * 0.12;
      renderScene();
    }
    animationFrame = requestAnimationFrame(animate);
  };
  animationFrame = requestAnimationFrame(animate);

  const updateActivity = (frame: BrainActivity | null) => {
    if (!geometry) return;
    for (const point of activePoints) activity[point] = 0;
    activePoints = [];
    if (frame?.signal === "emitted_simulated_spike_count") {
      for (const value of frame.values) {
        const point = pointByBodyId.get(value[1]);
        if (point === undefined || !Number.isFinite(value[2])) continue;
        activity[point] = Math.max(0, Math.min(1, value[2]));
        activePoints.push(point);
      }
    }
    geometry.getAttribute("activity").needsUpdate = true;
    repaint();
  };

  const setSelection = (inputs: MappedNeuron[], outputs: MappedNeuron[], watch: MappedNeuron[]) => {
    if (!geometry) {
      window.setTimeout(() => setSelection(inputs, outputs, watch), 100);
      return;
    }
    selection.fill(0);
    pointByNeuronIndex = new Map<number, number>();
    const apply = (items: MappedNeuron[], code: number) => {
      for (const item of items) {
        const point = pointByBodyId.get(item.body_id);
        if (point !== undefined) {
          selection[point] = code;
          pointByNeuronIndex.set(item.neuron_index, point);
        }
      }
    };
    apply(outputs, 2);
    apply(inputs, 1);
    apply(watch, 3);
    geometry.getAttribute("selection").needsUpdate = true;
    repaint();
  };

  const setNeighborhood = (value: Neighborhood | null) => {
    if (lines) {
      anatomy.remove(lines);
      lines.geometry.dispose();
      (lines.material as THREE.Material).dispose();
      lines = null;
    }
    if (!value || !geometry) { repaint(); return; }
    const atlasRowByIndex = new Map(value.nodes.map((node) => [node.neuron_index, node.atlas_row]));
    const vertices: number[] = [];
    const colors: number[] = [];
    const maximum = Math.max(1, ...value.edges.map((edge) => edge.synapse_count));
    for (const edge of value.edges) {
      const sourceRow = atlasRowByIndex.get(edge.source_neuron_index);
      const targetRow = atlasRowByIndex.get(edge.target_neuron_index);
      if (sourceRow === undefined || targetRow === undefined) continue;
      const sourcePoint = pointByAtlasRow[sourceRow];
      const targetPoint = pointByAtlasRow[targetRow];
      if (sourcePoint < 0 || targetPoint < 0) continue;
      for (const point of [sourcePoint, targetPoint]) {
        const offset = point * 3;
        vertices.push(
          transformedPositions[offset], transformedPositions[offset + 1],
          transformedPositions[offset + 2],
        );
      }
      const strength = Math.sqrt(edge.synapse_count / maximum);
      colors.push(0.15, 0.45 + strength * 0.4, 0.52 + strength * 0.45);
      colors.push(0.15, 0.45 + strength * 0.4, 0.52 + strength * 0.45);
    }
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
    lineGeometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    lines = new THREE.LineSegments(
      lineGeometry,
      new THREE.LineBasicMaterial({vertexColors: true, transparent: true, opacity: 0.55}),
    );
    anatomy.add(lines);
    repaint();
  };

  return {
    updateActivity,
    setSelection,
    setNeighborhood,
    resetView() { orbit = false; anatomy.rotation.set(0, 0, 0); fit(); },
    setOrbit(enabled: boolean) { orbit = enabled; },
    setRendering(enabled: boolean) { rendering = enabled; if (enabled) repaint(); },
    getPerformance() {
      return {
        render_count: renderCount,
        mean_cpu_submit_ms: renderCount ? renderCpuMs / renderCount : 0,
        max_cpu_submit_ms: maximumRenderCpuMs,
      };
    },
    dispose() {
      disposed = true;
      abort.abort();
      cancelAnimationFrame(animationFrame);
      observer.disconnect();
      renderer.domElement.removeEventListener("pointerdown", down);
      renderer.domElement.removeEventListener("pointermove", move);
      renderer.domElement.removeEventListener("pointerup", up);
      renderer.domElement.removeEventListener("pointercancel", up);
      if (lines) { lines.geometry.dispose(); (lines.material as THREE.Material).dispose(); }
      geometry?.dispose();
      material?.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      pointByBodyId.clear();
      pointByNeuronIndex.clear();
    },
  };
}
