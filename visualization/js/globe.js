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
    this.autoRotate = false;
    this.autoRotateSpeed = 0.0;

    this._initHolographicGlobe();
    this._initGraticuleRings();
  }

  _initHolographicGlobe() {
    // 1. Sleek Neutral Inner Core
    const coreGeometry = new THREE.SphereGeometry(this.radius * 0.99, 48, 48);
    const coreMaterial = new THREE.MeshBasicMaterial({
      color: 0x060911,
      transparent: true,
      opacity: 0.3,
      depthWrite: false,
    });
    this.coreMesh = new THREE.Mesh(coreGeometry, coreMaterial);
    this.globeGroup.add(this.coreMesh);

    // 2. Fine Precision Latitude & Longitude Wireframe Grid
    const wireGeo = new THREE.SphereGeometry(this.radius, 24, 18);
    const wireMat = new THREE.MeshBasicMaterial({
      color: 0x64748b,
      wireframe: true,
      transparent: true,
      opacity: 0.16,
    });
    this.wireMesh = new THREE.Mesh(wireGeo, wireMat);
    this.globeGroup.add(this.wireMesh);

    // 3. Crisp Spatial Particle Continents
    this._initProceduralContinents();
  }

  _initProceduralContinents() {
    const pointCount = 3600;
    const positions = new Float32Array(pointCount * 3);
    const colors = new Float32Array(pointCount * 3);

    const baseColor = new THREE.Color(0x94a3b8);
    const altColor = new THREE.Color(0x00f0ff);

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

      const mixed = baseColor.clone().lerp(altColor, Math.sin(phi * 4.0) * 0.4 + 0.1);
      colors[i * 3] = mixed.r;
      colors[i * 3 + 1] = mixed.g;
      colors[i * 3 + 2] = mixed.b;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.038,
      vertexColors: true,
      transparent: true,
      opacity: 0.75,
    });

    this.pointsMesh = new THREE.Points(geometry, material);
    this.globeGroup.add(this.pointsMesh);
  }

  _initGraticuleRings() {
    // Equator Ring
    const equatorGeo = new THREE.RingGeometry(this.radius * 1.005, this.radius * 1.012, 64);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x475569,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.35,
    });
    this.equatorRing = new THREE.Mesh(equatorGeo, ringMat);
    this.equatorRing.rotation.x = Math.PI / 2;
    this.globeGroup.add(this.equatorRing);
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
