/**
 * Interactive 3D Globe component built with Three.js.
 */

class InteractiveGlobe {
  constructor(scene, radius = 2.4) {
    this.scene = scene;
    this.radius = radius;

    this.globeGroup = new THREE.Group();
    this.scene.add(this.globeGroup);

    this.targetRotationY = 0;
    this.targetRotationX = 0;
    this.currentRotationY = 0;
    this.currentRotationX = 0;
    this.autoRotate = true;
    this.autoRotateSpeed = 0.0012;

    this._initHolographicGlobe();
    this._initGraticuleRings();
    this._initAtmosphereGlow();
  }

  _initHolographicGlobe() {
    // 1. Translucent Holographic Inner Core
    const coreGeometry = new THREE.SphereGeometry(this.radius * 0.99, 48, 48);
    const coreMaterial = new THREE.MeshBasicMaterial({
      color: 0x002244,
      transparent: true,
      opacity: 0.35,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.coreMesh = new THREE.Mesh(coreGeometry, coreMaterial);
    this.globeGroup.add(this.coreMesh);

    // 2. Holographic Latitude & Longitude Wireframe Cage
    const wireGeo = new THREE.SphereGeometry(this.radius, 24, 18);
    const wireMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      wireframe: true,
      transparent: true,
      opacity: 0.22,
      blending: THREE.AdditiveBlending,
    });
    this.wireMesh = new THREE.Mesh(wireGeo, wireMat);
    this.globeGroup.add(this.wireMesh);

    // 3. High-density Glowing Hologram Point Cloud Continents
    this._initProceduralContinents();
  }

  _initProceduralContinents() {
    const pointCount = 3600;
    const positions = new Float32Array(pointCount * 3);
    const colors = new Float32Array(pointCount * 3);

    const baseColor = new THREE.Color(0x00f0ff);
    const altColor = new THREE.Color(0x38bdf8);

    for (let i = 0; i < pointCount; i++) {
      const phi = Math.acos(1 - 2 * (i + 0.5) / pointCount);
      const theta = Math.PI * (1 + 5**0.5) * i;

      const r = this.radius * 1.002;
      const x = r * Math.sin(phi) * Math.cos(theta);
      const y = r * Math.cos(phi);
      const z = r * Math.sin(phi) * Math.sin(theta);

      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;

      const mixed = baseColor.clone().lerp(altColor, Math.sin(phi * 4.0) * 0.5 + 0.5);
      colors[i * 3] = mixed.r;
      colors[i * 3 + 1] = mixed.g;
      colors[i * 3 + 2] = mixed.b;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.045,
      vertexColors: true,
      transparent: true,
      opacity: 0.88,
      blending: THREE.AdditiveBlending,
    });

    this.pointsMesh = new THREE.Points(geometry, material);
    this.globeGroup.add(this.pointsMesh);
  }

  _initGraticuleRings() {
    // Equator Hologram Ring
    const equatorGeo = new THREE.RingGeometry(this.radius * 1.01, this.radius * 1.025, 64);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.45,
      blending: THREE.AdditiveBlending,
    });
    this.equatorRing = new THREE.Mesh(equatorGeo, ringMat);
    this.equatorRing.rotation.x = Math.PI / 2;
    this.globeGroup.add(this.equatorRing);

    // Orbital Telemetry Horizon Ring
    const orbitGeo = new THREE.RingGeometry(this.radius * 1.25, this.radius * 1.26, 64);
    const orbitMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.25,
      blending: THREE.AdditiveBlending,
    });
    this.orbitRing = new THREE.Mesh(orbitGeo, orbitMat);
    this.orbitRing.rotation.x = Math.PI / 3;
    this.globeGroup.add(this.orbitRing);
  }

  _initAtmosphereGlow() {
    // Outer Holographic Rim Glow Shell
    const glowGeo = new THREE.SphereGeometry(this.radius * 1.12, 48, 48);
    const glowMat = new THREE.ShaderMaterial({
      vertexShader: `
        varying vec3 vNormal;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        void main() {
          float intensity = pow(0.70 - dot(vNormal, vec3(0, 0, 1.0)), 2.5);
          gl_FragColor = vec4(0.0, 0.94, 1.0, 1.0) * intensity * 0.85;
        }
      `,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
    });

    this.atmosphereGlow = new THREE.Mesh(glowGeo, glowMat);
    this.globeGroup.add(this.atmosphereGlow);
  }

  applyRotationDelta(deltaYaw, deltaPitch) {
    this.autoRotate = false;
    this.targetRotationY += deltaYaw;
    this.targetRotationX += deltaPitch;

    // Pitch clamping to avoid gimbal flipping
    this.targetRotationX = Math.max(-Math.PI / 2.5, Math.min(Math.PI / 2.5, this.targetRotationX));
  }

  update(deltaTime) {
    if (this.autoRotate) {
      this.targetRotationY += this.autoRotateSpeed;
    }

    // Smooth spherical interpolation (lerp)
    const lerpFactor = 0.14;
    this.currentRotationY += (this.targetRotationY - this.currentRotationY) * lerpFactor;
    this.currentRotationX += (this.targetRotationX - this.currentRotationX) * lerpFactor;

    this.globeGroup.rotation.y = this.currentRotationY;
    this.globeGroup.rotation.x = this.currentRotationX;

    if (this.orbitRing) {
      this.orbitRing.rotation.z += deltaTime * 0.15;
    }
  }
}
