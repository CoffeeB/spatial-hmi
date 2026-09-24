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
    this.autoRotateSpeed = 0.0015;

    this._initGlobeMesh();
    this._initGraticuleRings();
    this._initAtmosphereGlow();
    this._initStarfield();
  }

  _initGlobeMesh() {
    // 1. Procedural Spherical Surface with custom shader material
    const sphereGeometry = new THREE.SphereGeometry(this.radius, 64, 64);

    // Deep space navy base with subtle Fresnel glow
    const sphereMaterial = new THREE.MeshStandardMaterial({
      color: 0x091428,
      metalness: 0.85,
      roughness: 0.35,
      emissive: 0x030814,
      emissiveIntensity: 0.6,
      wireframe: false,
    });

    this.sphereMesh = new THREE.Mesh(sphereGeometry, sphereMaterial);
    this.globeGroup.add(this.sphereMesh);

    // 2. High-contrast continent dot matrix overlay
    this._initProceduralContinents();
  }

  _initProceduralContinents() {
    const pointCount = 2800;
    const positions = new Float32Array(pointCount * 3);
    const colors = new Float32Array(pointCount * 3);

    const baseColor = new THREE.Color(0x00f0ff);
    const altColor = new THREE.Color(0x3b82f6);

    for (let i = 0; i < pointCount; i++) {
      // Golden spiral distribution across sphere
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
      size: 0.032,
      vertexColors: true,
      transparent: true,
      opacity: 0.75,
      blending: THREE.AdditiveBlending,
    });

    this.pointsMesh = new THREE.Points(geometry, material);
    this.globeGroup.add(this.pointsMesh);
  }

  _initGraticuleRings() {
    // Equator ring
    const equatorGeo = new THREE.RingGeometry(this.radius * 1.005, this.radius * 1.015, 64);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.25,
      wireframe: true,
    });
    const equator = new THREE.Mesh(equatorGeo, ringMat);
    equator.rotation.x = Math.PI / 2;
    this.globeGroup.add(equator);
  }

  _initAtmosphereGlow() {
    // Outer atmospheric glow shell
    const glowGeo = new THREE.SphereGeometry(this.radius * 1.15, 48, 48);
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
          float intensity = pow(0.65 - dot(vNormal, vec3(0, 0, 1.0)), 2.8);
          gl_FragColor = vec4(0.0, 0.94, 1.0, 1.0) * intensity * 0.7;
        }
      `,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
    });

    this.atmosphereGlow = new THREE.Mesh(glowGeo, glowMat);
    this.scene.add(this.atmosphereGlow);
  }

  _initStarfield() {
    const starCount = 1200;
    const starGeo = new THREE.BufferGeometry();
    const starPos = new Float32Array(starCount * 3);

    for (let i = 0; i < starCount * 3; i += 3) {
      starPos[i] = (Math.random() - 0.5) * 80;
      starPos[i + 1] = (Math.random() - 0.5) * 80;
      starPos[i + 2] = (Math.random() - 0.5) * 80;
    }

    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
    const starMat = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 0.08,
      transparent: true,
      opacity: 0.5,
    });

    const starfield = new THREE.Points(starGeo, starMat);
    this.scene.add(starfield);
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
    const lerpFactor = 0.12;
    this.currentRotationY += (this.targetRotationY - this.currentRotationY) * lerpFactor;
    this.currentRotationX += (this.targetRotationX - this.currentRotationX) * lerpFactor;

    this.globeGroup.rotation.y = this.currentRotationY;
    this.globeGroup.rotation.x = this.currentRotationX;
  }
}
