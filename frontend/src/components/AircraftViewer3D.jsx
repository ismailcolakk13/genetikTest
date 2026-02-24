import { useRef, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { TrackballControls, Text, Line } from "@react-three/drei";
import * as THREE from "three";

/* ===== Parametrik Gövde Mesh ===== */
function FuselageBody({ geometry }) {
  const mesh = useMemo(() => {
    if (!geometry) return null;

    const {
      fuselage_length: L,
      fuselage_width: W,
      fuselage_height: H,
      nose_length,
      mid_end,
    } = geometry;
    const halfW = W / 2;
    const halfH = H / 2;
    const longSegs = 64;
    const radSegs = 32;

    // Gövde profili fonksiyonu
    function profile(x) {
      if (x < 0 || x > L) return { ry: 0, rz: 0 };
      if (x < nose_length) {
        // Burun: ogive profil
        const t = x / nose_length;
        const r = 1 - (1 - t) * (1 - t); // Quadratic ease-in
        return { ry: halfW * r, rz: halfH * r * 0.85 }; // Burun daha yassı
      } else if (x < mid_end) {
        // Orta gövde: tam kesit
        return { ry: halfW, rz: halfH };
      } else {
        // Kuyruk: lineer daralma
        const t = (x - mid_end) / (L - mid_end);
        const r = 1 - t * 0.75;
        return { ry: halfW * r * 0.85, rz: halfH * r * 0.7 };
      }
    }

    const vertices = [];
    const indices = [];
    const normals = [];

    for (let i = 0; i <= longSegs; i++) {
      const x = (i / longSegs) * L;
      const { ry, rz } = profile(x);

      for (let j = 0; j <= radSegs; j++) {
        const angle = (j / radSegs) * Math.PI * 2;
        // Süper elips kesiti (savaş uçağı gövdesi daha köşeli)
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const n = 2.5; // Süper elips üssü (2 = elips, >2 = daha köşeli)
        const sc = Math.sign(cos) * Math.pow(Math.abs(cos), 2 / n);
        const ss = Math.sign(sin) * Math.pow(Math.abs(sin), 2 / n);

        const y = ry * sc;
        const z = rz * ss;

        vertices.push(x, y, z);

        // Normal hesabı (yaklaşık)
        const ny = sc / (ry || 1);
        const nz = ss / (rz || 1);
        const nl = Math.sqrt(ny * ny + nz * nz) || 1;
        normals.push(0, ny / nl, nz / nl);
      }
    }

    const cols = radSegs + 1;
    for (let i = 0; i < longSegs; i++) {
      for (let j = 0; j < radSegs; j++) {
        const a = i * cols + j;
        const b = a + 1;
        const c = (i + 1) * cols + j;
        const d = c + 1;
        indices.push(a, c, b, b, c, d);
      }
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(vertices, 3),
    );
    geom.setAttribute("normal", new THREE.Float32BufferAttribute(normals, 3));
    geom.setIndex(indices);
    geom.computeVertexNormals();
    return geom;
  }, [geometry]);

  if (!mesh) return null;

  return (
    <group>
      {/* Yarı şeffaf yüzey */}
      <mesh geometry={mesh}>
        <meshPhysicalMaterial
          color="#b0bec5"
          transparent
          opacity={0.18}
          side={THREE.DoubleSide}
          roughness={0.4}
          metalness={0.3}
          depthWrite={false}
        />
      </mesh>
      {/* Kenar vurgusu */}
      <mesh geometry={mesh}>
        <meshPhysicalMaterial
          color="#78909c"
          wireframe
          transparent
          opacity={0.04}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

/* ===== Gövde Kesit Çizgileri ===== */
function FuselageFrames({ geometry }) {
  const frames = useMemo(() => {
    if (!geometry) return [];

    const {
      fuselage_length: L,
      fuselage_width: W,
      fuselage_height: H,
      nose_length,
      mid_end,
    } = geometry;
    const halfW = W / 2;
    const halfH = H / 2;
    const n = 2.5;

    function profile(x) {
      if (x < nose_length) {
        const t = x / nose_length;
        const r = 1 - (1 - t) * (1 - t);
        return { ry: halfW * r, rz: halfH * r * 0.85 };
      } else if (x < mid_end) {
        return { ry: halfW, rz: halfH };
      } else {
        const t = (x - mid_end) / (L - mid_end);
        const r = 1 - t * 0.75;
        return { ry: halfW * r * 0.85, rz: halfH * r * 0.7 };
      }
    }

    // Kesit pozisyonları
    const positions = [];
    const numFrames = 12;
    for (let i = 0; i <= numFrames; i++) {
      positions.push((i / numFrames) * L);
    }
    // Burun ve kuyruk ekstra çizgileri
    positions.push(nose_length, mid_end);

    const result = [];
    for (const x of positions) {
      if (x < 0 || x > L) continue;
      const { ry, rz } = profile(x);
      if (ry < 1 || rz < 1) continue;

      const pts = [];
      for (let j = 0; j <= 48; j++) {
        const angle = (j / 48) * Math.PI * 2;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const sc = Math.sign(cos) * Math.pow(Math.abs(cos), 2 / n);
        const ss = Math.sign(sin) * Math.pow(Math.abs(sin), 2 / n);
        pts.push([x, ry * sc, rz * ss]);
      }
      result.push(pts);
    }

    return result;
  }, [geometry]);

  return (
    <>
      {frames.map((pts, i) => (
        <Line
          key={i}
          points={pts}
          color="#90a4ae"
          lineWidth={0.8}
          transparent
          opacity={0.4}
        />
      ))}
    </>
  );
}

/* ===== Kanat ===== */
function Wings({ geometry }) {
  if (!geometry?.wing) return null;

  const { wing, fuselage_width } = geometry;
  const {
    span,
    chord_root,
    chord_tip,
    position_x,
    position_z,
    sweep_angle_deg,
  } = wing;
  const halfSpan = span / 2;
  const halfBody = (fuselage_width / 2) * 0.9; // Gövde yarısı (kanat kökü)
  const sweepRad = (sweep_angle_deg * Math.PI) / 180;
  const thickness = chord_root * 0.06; // Kanat kalınlığı

  const wingGeom = useMemo(() => {
    // Kanat profili noktaları: kök ve uç
    // Kök: gövde kenarından başlar
    // Uç: span/2'ye kadar gider, süpürme uygulanır

    const spanLength = halfSpan - halfBody; // Kanat kök hariç uzunluk
    const tipSweepOffset = Math.tan(sweepRad) * spanLength;
    const tipX = position_x + tipSweepOffset;

    // Her kanat için tepe ve alt yüzey
    function makeWingHalf(sign) {
      // sign: -1 sol, +1 sağ
      const y_root = sign * halfBody;
      const y_tip = sign * halfSpan;

      // Leading edge (ön kenar)
      const le_root = [position_x, y_root, position_z];
      const le_tip = [tipX, y_tip, position_z];

      // Trailing edge (arka kenar)
      const te_root = [position_x + chord_root, y_root, position_z];
      const te_tip = [tipX + chord_tip, y_tip, position_z];

      // Kalınlık (orta noktada max)
      const mid_root_top = [
        position_x + chord_root * 0.35,
        y_root,
        position_z + thickness,
      ];
      const mid_root_bot = [
        position_x + chord_root * 0.35,
        y_root,
        position_z - thickness * 0.5,
      ];
      const mid_tip_top = [
        tipX + chord_tip * 0.35,
        y_tip,
        position_z + thickness * 0.3,
      ];
      const mid_tip_bot = [
        tipX + chord_tip * 0.35,
        y_tip,
        position_z - thickness * 0.15,
      ];

      // Basitleştirilmiş kanat: üst ve alt yüzey
      // Chord boyunca 5 nokta, span boyunca 8 dilim
      const spanDivs = 12;
      const chordDivs = 8;

      const verts = [];
      const idxs = [];

      for (let s = 0; s <= spanDivs; s++) {
        const st = s / spanDivs;
        const y = y_root + (y_tip - y_root) * st;
        const leX = position_x + tipSweepOffset * st;
        const chord = chord_root + (chord_tip - chord_root) * st;
        const localThick = thickness * (1 - st * 0.7);

        for (let c = 0; c <= chordDivs; c++) {
          const ct = c / chordDivs;
          const x = leX + chord * ct;

          // NACA benzeri profil (üst yüzey)
          const t_naca = localThick * 2;
          const z_camber =
            t_naca *
            (0.2969 * Math.sqrt(ct) -
              0.126 * ct -
              0.3516 * ct * ct +
              0.2843 * ct * ct * ct -
              0.1015 * ct * ct * ct * ct);

          verts.push(x, y, position_z + z_camber);
        }
      }

      // Alt yüzey
      const offset = verts.length / 3;
      for (let s = 0; s <= spanDivs; s++) {
        const st = s / spanDivs;
        const y = y_root + (y_tip - y_root) * st;
        const leX = position_x + tipSweepOffset * st;
        const chord = chord_root + (chord_tip - chord_root) * st;
        const localThick = thickness * (1 - st * 0.7);

        for (let c = 0; c <= chordDivs; c++) {
          const ct = c / chordDivs;
          const x = leX + chord * ct;

          const t_naca = localThick * 2;
          const z_camber =
            -t_naca *
            (0.2969 * Math.sqrt(ct) -
              0.126 * ct -
              0.3516 * ct * ct +
              0.2843 * ct * ct * ct -
              0.1015 * ct * ct * ct * ct) *
            0.5;

          verts.push(x, y, position_z + z_camber);
        }
      }

      // Üst yüzey indeksleri
      const cols = chordDivs + 1;
      for (let s = 0; s < spanDivs; s++) {
        for (let c = 0; c < chordDivs; c++) {
          const a = s * cols + c;
          const b = a + 1;
          const d = (s + 1) * cols + c;
          const e = d + 1;
          idxs.push(a, d, b, b, d, e);
        }
      }

      // Alt yüzey indeksleri (ters yönde)
      for (let s = 0; s < spanDivs; s++) {
        for (let c = 0; c < chordDivs; c++) {
          const a = offset + s * cols + c;
          const b = a + 1;
          const d = offset + (s + 1) * cols + c;
          const e = d + 1;
          idxs.push(a, b, d, b, e, d);
        }
      }

      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
      geom.setIndex(idxs);
      geom.computeVertexNormals();
      return geom;
    }

    return { left: makeWingHalf(-1), right: makeWingHalf(1) };
  }, [geometry]);

  return (
    <group>
      <mesh geometry={wingGeom.left}>
        <meshPhysicalMaterial
          color="#78909c"
          transparent
          opacity={0.45}
          side={THREE.DoubleSide}
          roughness={0.3}
          metalness={0.5}
        />
      </mesh>
      <mesh geometry={wingGeom.right}>
        <meshPhysicalMaterial
          color="#78909c"
          transparent
          opacity={0.45}
          side={THREE.DoubleSide}
          roughness={0.3}
          metalness={0.5}
        />
      </mesh>
    </group>
  );
}

/* ===== Yatay Kuyruk ===== */
function HorizontalStabilizer({ geometry }) {
  if (!geometry?.wing) return null;

  const L = geometry.fuselage_length;
  const tailX = L * 0.84;
  const hSpan = geometry.wing.span * 0.28;
  const chordRoot = geometry.wing.chord_root * 0.35;
  const chordTip = chordRoot * 0.4;
  const sweepOffset = chordRoot * 0.8;

  const geom = useMemo(() => {
    const halfBody = geometry.fuselage_width * 0.15;

    function makeHalf(sign) {
      const yRoot = sign * halfBody;
      const yTip = (sign * hSpan) / 2;

      const verts = [
        tailX,
        yRoot,
        0,
        tailX + chordRoot,
        yRoot,
        0,
        tailX + sweepOffset + chordTip,
        yTip,
        0,
        tailX + sweepOffset,
        yTip,
        0,
      ];

      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
      geom.setIndex([0, 1, 2, 0, 2, 3]);
      geom.computeVertexNormals();
      return geom;
    }

    return { left: makeHalf(-1), right: makeHalf(1) };
  }, [geometry]);

  return (
    <group>
      <mesh geometry={geom.left}>
        <meshPhysicalMaterial
          color="#78909c"
          transparent
          opacity={0.45}
          side={THREE.DoubleSide}
          roughness={0.3}
          metalness={0.5}
        />
      </mesh>
      <mesh geometry={geom.right}>
        <meshPhysicalMaterial
          color="#78909c"
          transparent
          opacity={0.45}
          side={THREE.DoubleSide}
          roughness={0.3}
          metalness={0.5}
        />
      </mesh>
    </group>
  );
}

/* ===== Dikey Kuyruk (Canted) ===== */
function VerticalStabilizer({ geometry }) {
  if (!geometry?.wing) return null;

  const L = geometry.fuselage_length;
  const vs = geometry?.vertical_stabilizer || {};
  const cantAngle = ((vs.cant_angle_deg || 25) * Math.PI) / 180;
  const height = vs.height || L * 0.13;
  const chordRoot = vs.chord_root || L * 0.12;
  const chordTip = vs.chord_tip || chordRoot * 0.45;
  const tailX = vs.position_x || L * 0.82;
  const sweepOffset = chordRoot * 0.5;
  const yOffset = geometry.fuselage_width * 0.15;

  const geom = useMemo(() => {
    function makeFin(sign) {
      const yBase = sign * yOffset;
      const yTop = yBase + sign * Math.sin(cantAngle) * height;
      const zTop = Math.cos(cantAngle) * height;

      const verts = [
        tailX,
        yBase,
        0,
        tailX + chordRoot,
        yBase,
        0,
        tailX + sweepOffset + chordTip,
        yTop,
        zTop,
        tailX + sweepOffset,
        yTop,
        zTop,
      ];

      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
      g.setIndex([0, 1, 2, 0, 2, 3]);
      g.computeVertexNormals();
      return g;
    }
    return { left: makeFin(-1), right: makeFin(1) };
  }, [geometry]);

  return (
    <group>
      <mesh geometry={geom.left}>
        <meshPhysicalMaterial
          color="#6b8fa3"
          transparent
          opacity={0.5}
          side={THREE.DoubleSide}
          roughness={0.3}
          metalness={0.5}
        />
      </mesh>
      <mesh geometry={geom.right}>
        <meshPhysicalMaterial
          color="#6b8fa3"
          transparent
          opacity={0.5}
          side={THREE.DoubleSide}
          roughness={0.3}
          metalness={0.5}
        />
      </mesh>
    </group>
  );
}

/* ===== Hava Girişleri (İtki) ===== */
function Intakes({ geometry }) {
  if (!geometry?.wing) return null;

  const intakeX = geometry.mid_start || geometry.nose_length;
  const intakeLen = geometry.fuselage_width * 0.6;
  const intakeW = geometry.fuselage_width * 0.15;
  const intakeH = geometry.fuselage_height * 0.12;
  const yOff = geometry.fuselage_width * 0.35;

  return (
    <group>
      {[-1, 1].map((sign) => (
        <mesh
          key={sign}
          position={[
            intakeX + intakeLen / 2,
            sign * yOff,
            -geometry.fuselage_height * 0.15,
          ]}
        >
          <boxGeometry args={[intakeLen, intakeW, intakeH]} />
          <meshPhysicalMaterial
            color="#455a64"
            transparent
            opacity={0.35}
            roughness={0.5}
            metalness={0.4}
          />
        </mesh>
      ))}
    </group>
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
          opacity={selected ? 0.92 : 0.8}
          roughness={0.2}
          metalness={0.4}
          emissive={selected ? color : "#000000"}
          emissiveIntensity={selected ? 0.3 : 0}
        />
      </mesh>
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(dx, dy, dz)]} />
        <lineBasicMaterial color={selected ? "#1e293b" : "#37474f"} />
      </lineSegments>
      <Text
        position={[0, 0, dz / 2 + 8]}
        fontSize={Math.min(10, dx * 0.18)}
        color="#1e293b"
        anchorX="center"
        anchorY="bottom"
        outlineWidth={0.4}
        outlineColor="white"
      >
        {name}
      </Text>
    </group>
  );
}

/* ===== CG Hedef Bölgesi ===== */
function CGTargetZone({ cgTarget, fuselageWidth, fuselageHeight }) {
  if (!cgTarget) return null;
  const { target_x_min, target_x_max } = cgTarget;
  const length = target_x_max - target_x_min;
  const centerX = (target_x_min + target_x_max) / 2;
  const boxW = fuselageWidth * 0.7;
  const boxH = (fuselageHeight || fuselageWidth) * 0.7;

  return (
    <mesh position={[centerX, 0, 0]}>
      <boxGeometry args={[length, boxW, boxH]} />
      <meshBasicMaterial
        color="#059669"
        transparent
        opacity={0.06}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

/* ===== CG Noktası ===== */
function CGMarker({ position, fuselageLength }) {
  if (!position) return null;
  const [x, y, z] = position;
  const markerSize = (fuselageLength || 1000) * 0.008;

  return (
    <group>
      <mesh position={[x, y, z]}>
        <sphereGeometry args={[markerSize, 16, 16]} />
        <meshPhysicalMaterial
          color="#dc2626"
          emissive="#dc2626"
          emissiveIntensity={0.6}
        />
      </mesh>
      <Line
        points={[
          [x, y, z - fuselageLength * 0.08],
          [x, y, z + fuselageLength * 0.08],
        ]}
        color="#dc2626"
        lineWidth={1.5}
        dashed
        dashSize={5}
        gapSize={5}
      />
      <Text
        position={[x, y, z + fuselageLength * 0.09]}
        fontSize={fuselageLength * 0.014}
        color="#dc2626"
        anchorX="center"
        outlineWidth={0.4}
        outlineColor="white"
        fontWeight="bold"
      >
        CG
      </Text>
    </group>
  );
}

/* ===== Grid ===== */
function GroundGrid({ length }) {
  return (
    <gridHelper
      args={[length * 2, 40, "#cbd5e1", "#e8ecf0"]}
      position={[length / 2, 0, -length * 0.15]}
    />
  );
}

/* ===== Eksen Çizgileri ===== */
function AxisLines({ length }) {
  return (
    <group>
      <Line
        points={[
          [0, 0, 0],
          [length, 0, 0],
        ]}
        color="#94a3b8"
        lineWidth={1}
        dashed
        dashSize={length * 0.02}
        gapSize={length * 0.01}
      />
    </group>
  );
}

/* ===== Kategori Renkleri ===== */
const CATEGORY_COLORS = {
  propulsion: "#0891b2",
  avionics: "#7c3aed",
  fuel: "#d97706",
  landing_gear: "#475569",
  weapon: "#dc2626",
  systems: "#059669",
  electrical: "#2563eb",
  structure: "#8b5cf6",
  payload: "#db2777",
  general: "#64748b",
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
        <p style={{ color: "#94a3b8", fontSize: 14 }}>Uçak tipi seçiniz</p>
      </div>
    );
  }

  const { geometry, components, cg_target } = aircraftData;
  const fL = geometry.fuselage_length;

  return (
    <div className="viewer-container">
      <div className="viewer-overlay">
        <span className="viewer-badge">🎯 3D Görüntüleyici</span>
        <span className="viewer-badge">
          Fare ile döndür · Scroll ile yakınlaş
        </span>
      </div>

      <Canvas
        camera={{
          position: [fL * 0.6, -fL * 0.5, fL * 0.35],
          fov: 45,
          near: 1,
          far: fL * 10,
        }}
        style={{ background: "#f0f4f8" }}
      >
        {/* Işıklandırma */}
        <ambientLight intensity={0.5} />
        <directionalLight position={[fL, -fL * 0.5, fL]} intensity={0.7} />
        <directionalLight
          position={[-fL * 0.5, fL, fL * 0.3]}
          intensity={0.3}
        />
        <hemisphereLight
          skyColor="#b3d4fc"
          groundColor="#fce4ec"
          intensity={0.3}
        />

        <TrackballControls
          target={[fL * 0.45, 0, 0]}
          rotateSpeed={3}
          zoomSpeed={2}
          panSpeed={1}
          noRotate={false}
          noPan={false}
          noZoom={false}
          staticMoving={false}
          dynamicDampingFactor={0.1}
        />

        {/* Uçak Geometrisi */}
        <FuselageBody geometry={geometry} />
        <FuselageFrames geometry={geometry} />
        <Wings geometry={geometry} />
        <HorizontalStabilizer geometry={geometry} />
        <VerticalStabilizer geometry={geometry} />
        <Intakes geometry={geometry} />

        {/* CG Hedef Bölgesi */}
        <CGTargetZone
          cgTarget={cg_target}
          fuselageWidth={geometry.fuselage_width}
          fuselageHeight={geometry.fuselage_height}
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
        <CGMarker position={cgPosition} fuselageLength={fL} />

        {/* Grid & Eksen */}
        <GroundGrid length={fL} />
        <AxisLines length={fL} />
      </Canvas>
    </div>
  );
}
