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
    this.targetRotationZ = 0;
    this.currentRotationY = 0;
    this.currentRotationX = 0;
    this.currentRotationZ = 0;

    this.targetPosX = 0;
    this.targetPosY = 0;
    this.currentPosX = 0;
    this.currentPosY = 0;

    this.autoRotate = false;
    this.autoRotateSpeed = 0.0;

    // Inertia: velocity used to damp motion when hands are released
    this._velY = 0;
    this._velX = 0;
    this._frozen = false;

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
    this.boundedYaw = 0;
    this.boundedPitch = 0;
    this.boundedRoll = 0;
    this._swipeVelY = 0;
    this._swipeVelX = 0;
  }

  applySwipeSpin(deltaYaw, deltaPitch) {
    this.autoRotate = false;
    this._frozen = false;
    // Gradual, cinematic 360-degree spin impulse
    const maxImpulse = 0.08;
    this._swipeVelY = Math.max(-maxImpulse, Math.min(maxImpulse, deltaYaw * 0.016));
    this._swipeVelX = Math.max(-maxImpulse, Math.min(maxImpulse, deltaPitch * 0.016));
  }

  applyBoundedHandRotation(deltaYaw, deltaPitch, deltaRoll = 0) {
    this.autoRotate = false;
    this._frozen = false;
    this._swipeVelY = 0;
    this._swipeVelX = 0;

    // Direct gradual bounded manipulation: edge-to-edge horizontally and top-to-bottom vertically
    const scale = 0.55;
    this.boundedYaw = Math.max(-Math.PI * 0.95, Math.min(Math.PI * 0.95, this.boundedYaw + deltaYaw * scale));
    this.boundedPitch = Math.max(-Math.PI / 2.2, Math.min(Math.PI / 2.2, this.boundedPitch + deltaPitch * scale));
    this.boundedRoll = Math.max(-Math.PI / 3.0, Math.min(Math.PI / 3.0, this.boundedRoll + deltaRoll * scale));

    this.targetRotationY = this.boundedYaw;
    this.targetRotationX = this.boundedPitch;
    this.targetRotationZ = this.boundedRoll;
  }

  applyRotationDelta(deltaYaw, deltaPitch, deltaRoll = 0) {
    this.autoRotate = false;
    this._frozen = false;
    const scale = 0.60;
    this.targetRotationY += deltaYaw * scale;
    this.targetRotationX += deltaPitch * scale;
    this.targetRotationZ += deltaRoll * scale;
    this._velY = deltaYaw * scale;
    this._velX = deltaPitch * scale;

    // Pitch clamping to avoid gimbal flipping
    this.targetRotationX = Math.max(-Math.PI / 2.5, Math.min(Math.PI / 2.5, this.targetRotationX));
    this.boundedYaw = this.targetRotationY;
    this.boundedPitch = this.targetRotationX;
  }

  applyTranslationDelta(dx, dy) {
    this._frozen = false;
    this.targetPosX += dx;
    this.targetPosY += dy;
    // Clamp translation limits so globe remains visible
    this.targetPosX = Math.max(-3.5, Math.min(3.5, this.targetPosX));
    this.targetPosY = Math.max(-2.5, Math.min(2.5, this.targetPosY));
  }

  /**
   * Freeze all motion immediately — called when hands drop or IDLE state is entered.
   * Snaps target to current position so the lerp doesn't continue chasing a stale target.
   */
  freeze() {
    this._frozen = true;
    this._velY = 0;
    this._velX = 0;
    this._swipeVelY = 0;
    this._swipeVelX = 0;
    // Snap targets to current to stop lerp motion
    this.targetRotationY = this.currentRotationY;
    this.targetRotationX = this.currentRotationX;
    this.targetRotationZ = this.currentRotationZ;
    this.targetPosX = this.currentPosX;
    this.targetPosY = this.currentPosY;
    this.boundedYaw = this.currentRotationY;
    this.boundedPitch = this.currentRotationX;
    this.boundedRoll = this.currentRotationZ;
  }

  update(deltaTime) {
    if (this.autoRotate) {
      this.targetRotationY += this.autoRotateSpeed;
    }

    if (this._frozen) {
      // Globe is frozen — hold current position, no lerp drift
      this.globeGroup.rotation.y = this.currentRotationY;
      this.globeGroup.rotation.x = this.currentRotationX;
      this.globeGroup.rotation.z = this.currentRotationZ;
      this.globeGroup.position.x = this.currentPosX;
      this.globeGroup.position.y = this.currentPosY;
      if (this.orbitRing) this.orbitRing.rotation.z += deltaTime * 0.15;
      return;
    }

    // Process swipe 360-degree spin physics
    if (Math.abs(this._swipeVelY) > 0.0002 || Math.abs(this._swipeVelX) > 0.0002) {
      this.currentRotationY += this._swipeVelY;
      this.targetRotationY = this.currentRotationY;
      this.boundedYaw = this.currentRotationY;
      this.currentRotationX += this._swipeVelX;
      this.targetRotationX = this.currentRotationX;
      this.boundedPitch = this.currentRotationX;
      this._swipeVelY *= 0.965; // Gradual smooth angular deceleration
      this._swipeVelX *= 0.965;
    } else {
      // Smooth gradual spherical interpolation towards bounded target
      const lerpFactor = 0.07;
      this.currentRotationY += (this.targetRotationY - this.currentRotationY) * lerpFactor;
      this.currentRotationX += (this.targetRotationX - this.currentRotationX) * lerpFactor;
      this.currentRotationZ += (this.targetRotationZ - this.currentRotationZ) * lerpFactor;
    }

    const lerpFactor = 0.07;
    this.currentPosX += (this.targetPosX - this.currentPosX) * lerpFactor;
    this.currentPosY += (this.targetPosY - this.currentPosY) * lerpFactor;

    this.globeGroup.rotation.y = this.currentRotationY;
    this.globeGroup.rotation.x = this.currentRotationX;
    this.globeGroup.rotation.z = this.currentRotationZ;
    this.globeGroup.position.x = this.currentPosX;
    this.globeGroup.position.y = this.currentPosY;

    if (this.orbitRing) {
      this.orbitRing.rotation.z += deltaTime * 0.15;
    }
  }
}
