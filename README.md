# 🚀 EvoBrainNet: A Multi-Objective Evolutionary Neural Architecture Search with Self-Adaptive Mutation for Volumetric Brain Tumor Segmentation

🚀 **Accurate, Efficient 3D Brain Tumor Segmentation in MRI using Evolutionary Architecture Search**

📌 **Full code and pretrained models will be released soon!**

---

## 🔍 Abstract

Precise segmentation of brain tumors in volumetric MRI is challenging due to significant heterogeneity in tumor shape, size, and intensity. **EvoBrainNet** tackles this by combining:

- **ExoFeature Module:** Enhanced contextual encoding for robust feature extraction.
- **Dilated Residual Attention Pyramid (DRAP):** Multiscale residual attention and channel recalibration.
- **RefineUp Module:** Decoder-side refinement with attention-guided upsampling.

A multi-objective evolutionary neural architecture search (NAS) framework—with a self-adaptive mutation strategy—jointly optimizes both segmentation quality (Dice similarity coefficient) and model efficiency (parameters, GFLOPs). EvoBrainNet outperforms nine state-of-the-art methods on multiple benchmarks and generalizes well across unseen datasets.

---

## 🎯 Key Features

✅ **State-of-the-Art Accuracy:** Achieves 95.56% Dice and 1.42mm HD95 on BraTS 2021.

✅ **Multi-Objective Optimization:** Simultaneously maximizes accuracy and efficiency.

✅ **Self-Adaptive Evolutionary NAS:** Automatically explores optimal architectures with dynamic mutation rate.

✅ **Generalization:** Robust performance on BraTS 2020 and MSD Brain Tumor datasets.

✅ **Ablation Proven:** Each module’s effectiveness is statistically validated.

---

## 🏆 Results at a Glance

| Dataset        | DSC (%) | HD95 (mm) |  
|----------------|---------|-----------|  
| **BraTS 2021** | 95.56   | 1.42      |  
| **BraTS 2020** | 93.08   | 1.97      |  
| **MSD Brain**  | 93.79   | 1.64      |  

*Outperforms 9 SOTA methods in both accuracy and efficiency.*

---

## 🧠 Core Innovations

- **Modular 3D Supernet:** Flexible, scalable architecture search space.
- **Self-Adaptive Mutation:** Dynamic evolution based on real-time performance.
- **Clinical Efficiency:** Optimized for parameter count and computational cost (GFLOPs).

---

## 📦 Stay Tuned

- 🔜 Full code, pretrained models, and detailed usage instructions coming soon!
- 💡 Open source for the research and clinical community.

