import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Text, Line } from "@react-three/drei";
import * as THREE from "three";

/* ===== Parametrik Gövde Mesh ===== */
function FuselageBody({ geometry }) {
  const meshRef = useRef();

  const fuselageGeom = useMemo(() => {
    if (!geometry) return null;

    const {
      fuselage_length,
      fuselage_width,
      fuselage_height,
      nose_length,
      mid_end,
    } = geometry;
    const halfW = fuselage_width / 2;
    const halfH = fuselage_height / 2;
    const segments = 48;
    const radialSegments = 24;

    const points = [];

    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      const x = t * fuselage_length;

      let radius;
      if (x < nose_length) {
        // Burun
        const r = x / nose_length;
        radius = Math.pow(r, 0.6);
      } else if (x < mid_end) {
        // Orta gövde
        radius = 1.0;
      } else {
        // Kuyruk
        const tailLen = fuselage_length - mid_end;
        const r = (x - mid_end) / (tailLen || 1);
        radius = 1.0 - r * 0.85;
      }

      for (let j = 0; j <= radialSegments; j++) {
        const angle = (j / radialSegments) * Math.PI * 2;
        const y = Math.cos(angle) * halfW * radius;
        const z = Math.sin(angle) * halfH * radius;
        points.push(new THREE.Vector3(x, y, z));
      }
    }

    const geom = new THREE.BufferGeometry();
    const vertices = [];
    const indices = [];

    for (const p of points) {
      vertices.push(p.x, p.y, p.z);
    }

    const cols = radialSegments + 1;
    for (let i = 0; i < segments; i++) {
      for (let j = 0; j < radialSegments; j++) {
        const a = i * cols + j;
        const b = a + 1;
        const c = (i + 1) * cols + j;
        const d = c + 1;
        indices.push(a, c, b);
        indices.push(b, c, d);
      }
    }

    geom.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(vertices, 3),
    );
    geom.setIndex(indices);
    geom.computeVertexNormals();

    return geom;
  }, [geometry]);

  if (!fuselageGeom) return null;

  return (
    <mesh ref={meshRef} geometry={fuselageGeom}>
      <meshPhysicalMaterial
        color="#6b7280"
        transparent
        opacity={0.12}
        side={THREE.DoubleSide}
        roughness={0.3}
        metalness={0.8}
      />
    </mesh>
  );
}

/* ===== Gövde Wireframe ===== */
function FuselageWireframe({ geometry }) {
  const lines = useMemo(() => {
    if (!geometry) return [];

    const {
      fuselage_length,
      fuselage_width,
      fuselage_height,
      nose_length,
      mid_end,
    } = geometry;
    const halfW = fuselage_width / 2;
    const halfH = fuselage_height / 2;
    const result = [];

    // Boylamasına kesit çizgileri
    const crossSections = [
      0,
      nose_length * 0.5,
      nose_length,
      mid_end * 0.5,
      mid_end,
      (mid_end + fuselage_length) * 0.5,
      fuselage_length * 0.9,
    ];

    for (const x of crossSections) {
      let radius;
      if (x < nose_length) {
        radius = Math.pow(x / nose_length, 0.6);
      } else if (x < mid_end) {
        radius = 1.0;
      } else {
        const tailLen = fuselage_length - mid_end;
        const r = (x - mid_end) / (tailLen || 1);
        radius = 1.0 - r * 0.85;
      }

      const pts = [];
      for (let j = 0; j <= 32; j++) {
        const angle = (j / 32) * Math.PI * 2;
        pts.push([
          x,
          Math.cos(angle) * halfW * radius,
          Math.sin(angle) * halfH * radius,
        ]);
      }
      result.push(pts);
    }

    return result;
  }, [geometry]);

  return (
    <>
      {lines.map((pts, i) => (
        <Line key={i} points={pts} color="#374151" lineWidth={0.5} />
      ))}
    </>
  );
}

/* ===== Kanat Mesh ===== */
function Wing({ geometry }) {
  if (!geometry?.wing) return null;

  const { wing } = geometry;
  const {
    span,
    chord_root,
    chord_tip,
    position_x,
    position_z,
    sweep_angle_deg,
  } = wing;
  const halfSpan = span / 2;
  const sweepOffset = Math.tan((sweep_angle_deg * Math.PI) / 180) * halfSpan;

  const shape = useMemo(() => {
    const vertices = new Float32Array([
      // Sol kanat
      position_x,
      -halfSpan,
      position_z,
      position_x + chord_root,
      -halfSpan * 0.12,
      position_z,
      position_x + sweepOffset + chord_tip,
      -halfSpan,
      position_z,
      position_x + sweepOffset,
      -halfSpan,
      position_z,
      // Sağ kanat
      position_x,
      halfSpan,
      position_z,
      position_x + chord_root,
      halfSpan * 0.12,
      position_z,
      position_x + sweepOffset + chord_tip,
      halfSpan,
      position_z,
      position_x + sweepOffset,
      halfSpan,
      position_z,
      // Gövde bağlantısı sol
      position_x,
      -halfSpan * 0.12,
      position_z,
      position_x + chord_root,
      -halfSpan * 0.12,
      position_z,
      // Gövde bağlantısı sağ
      position_x,
      halfSpan * 0.12,
      position_z,
      position_x + chord_root,
      halfSpan * 0.12,
      position_z,
    ]);

    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(vertices, 3));

    const indices = [
      0,
      1,
      2,
      0,
      2,
      3, // Sol kanat üst
      8,
      9,
      1,
      8,
      1,
      0, // Sol gövde bağlantısı
      4,
      5,
      6,
      4,
      6,
      7, // Sağ kanat üst
      10,
      11,
      5,
      10,
      5,
      4, // Sağ gövde bağlantısı
    ];
    geom.setIndex(indices);
    geom.computeVertexNormals();
    return geom;
  }, [geometry]);

  return (
    <mesh geometry={shape}>
      <meshPhysicalMaterial
        color="#60a5fa"
        transparent
        opacity={0.3}
        side={THREE.DoubleSide}
        roughness={0.4}
        metalness={0.6}
      />
    </mesh>
  );
}

/* ===== Komponent Kutusu ===== */
function ComponentBox({ position, size, color, name, selected, onClick }) {
  const meshRef = useRef();

  const [x, y, z] = position;
  const [dx, dy, dz] = size;

  return (
    <group position={[x, y, z]} onClick={onClick}>
      <mesh ref={meshRef}>
        <boxGeometry args={[dx, dy, dz]} />
        <meshPhysicalMaterial
          color={color}
          transparent
          opacity={selected ? 0.9 : 0.75}
          roughness={0.2}
          metalness={0.5}
          emissive={selected ? color : "#000000"}
          emissiveIntensity={selected ? 0.3 : 0}
        />
      </mesh>
      {/* Kenar çizgileri */}
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(dx, dy, dz)]} />
        <lineBasicMaterial
          color={selected ? "#ffffff" : "#000000"}
          linewidth={1}
        />
      </lineSegments>
      {/* İsim etiketi */}
      <Text
        position={[0, 0, dz / 2 + 8]}
        fontSize={10}
        color="white"
        anchorX="center"
        anchorY="bottom"
        outlineWidth={0.5}
        outlineColor="black"
      >
        {name}
      </Text>
    </group>
  );
}

/* ===== CG Hedef Bölgesi ===== */
function CGTargetZone({ cgTarget, fuselageWidth }) {
  if (!cgTarget) return null;

  const { target_x_min, target_x_max } = cgTarget;
  const boxW = fuselageWidth * 0.8;
  const boxH = fuselageWidth * 0.8;
  const length = target_x_max - target_x_min;
  const centerX = (target_x_min + target_x_max) / 2;

  return (
    <mesh position={[centerX, 0, 0]}>
      <boxGeometry args={[length, boxW, boxH]} />
      <meshBasicMaterial
        color="#10b981"
        transparent
        opacity={0.08}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

/* ===== CG Noktası ===== */
function CGMarker({ position }) {
  if (!position) return null;

  const [x, y, z] = position;

  return (
    <group>
      {/* CG noktası */}
      <mesh position={[x, y, z]}>
        <sphereGeometry args={[8, 16, 16]} />
        <meshPhysicalMaterial
          color="#ef4444"
          emissive="#ef4444"
          emissiveIntensity={0.5}
        />
      </mesh>
      {/* Dikey çizgi */}
      <Line
        points={[
          [x, y, z - 100],
          [x, y, z + 100],
        ]}
        color="#ef4444"
        lineWidth={1}
        dashed
        dashSize={5}
        gapSize={5}
      />
      {/* Etiket */}
      <Text
        position={[x, y, z + 115]}
        fontSize={14}
        color="#ef4444"
        anchorX="center"
        outlineWidth={0.5}
        outlineColor="black"
        fontWeight="bold"
      >
        CG
      </Text>
    </group>
  );
}

/* ===== Ana Grid ===== */
function GroundGrid({ length }) {
  return (
    <gridHelper
      args={[length * 1.5, 30, "#1e293b", "#111827"]}
      position={[length / 2, 0, -150]}
      rotation={[0, 0, 0]}
    />
  );
}

/* ===== Kategori Renkleri ===== */
const CATEGORY_COLORS = {
  propulsion: "#06b6d4",
  avionics: "#8b5cf6",
  fuel: "#f59e0b",
  landing_gear: "#6b7280",
  weapon: "#ef4444",
  systems: "#10b981",
  electrical: "#3b82f6",
  structure: "#a78bfa",
  payload: "#f472b6",
  general: "#94a3b8",
};

/* ===== Ana 3D Sahne ===== */
export default function AircraftViewer3D({
  aircraftData,
  layout,
  cgPosition,
  selectedComponent,
  onComponentClick,
}) {
  if (!aircraftData) {
    return (
      <div
        className="viewer-container"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <p style={{ color: "#64748b" }}>Uçak tipi seçiniz</p>
      </div>
    );
  }

  const { geometry, components, cg_target } = aircraftData;
  const fLength = geometry.fuselage_length;

  return (
    <div className="viewer-container">
      <div className="viewer-overlay">
        <span className="viewer-badge">🎯 3D Görüntüleyici</span>
        <span className="viewer-badge">
          Fare ile döndür | Scroll ile yakınlaş
        </span>
      </div>

      <Canvas
        camera={{
          position: [fLength * 0.8, -fLength * 0.6, fLength * 0.4],
          fov: 50,
          near: 1,
          far: fLength * 10,
        }}
        style={{ background: "#0a0e1a" }}
      >
        {/* Işıklandırma */}
        <ambientLight intensity={0.4} />
        <directionalLight
          position={[fLength, -fLength, fLength]}
          intensity={0.8}
        />
        <directionalLight
          position={[-fLength, fLength, -fLength / 2]}
          intensity={0.3}
        />
        <pointLight
          position={[fLength / 2, 0, fLength / 2]}
          intensity={0.5}
          color="#3b82f6"
        />

        {/* Kontroller */}
        <OrbitControls
          target={[fLength / 2, 0, 0]}
          enableDamping
          dampingFactor={0.05}
          minDistance={fLength * 0.3}
          maxDistance={fLength * 3}
        />

        {/* Gövde */}
        <FuselageBody geometry={geometry} />
        <FuselageWireframe geometry={geometry} />

        {/* Kanatlar */}
        <Wing geometry={geometry} />

        {/* CG Hedef Bölgesi */}
        <CGTargetZone
          cgTarget={cg_target}
          fuselageWidth={geometry.fuselage_width}
        />

        {/* Komponentler */}
        {components &&
          layout &&
          components.map((comp) => {
            const pos = layout[comp.id];
            if (!pos) return null;

            return (
              <ComponentBox
                key={comp.id}
                position={pos}
                size={comp.size}
                color={
                  CATEGORY_COLORS[comp.category] || CATEGORY_COLORS.general
                }
                name={comp.name || comp.id}
                selected={selectedComponent === comp.id}
                onClick={(e) => {
                  e.stopPropagation();
                  onComponentClick?.(comp.id);
                }}
              />
            );
          })}

        {/* CG Marker */}
        <CGMarker position={cgPosition} />

        {/* Grid */}
        <GroundGrid length={fLength} />
      </Canvas>
    </div>
  );
}
