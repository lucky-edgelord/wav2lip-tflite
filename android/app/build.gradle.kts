plugins { id("com.android.application") }

android {
    namespace = "dev.luckyedgelord.wav2lipbench"
    compileSdk = 35
    defaultConfig {
        applicationId = "dev.luckyedgelord.wav2lipbench"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += listOf("arm64-v8a") }
    }
    buildTypes {
        release { isMinifyEnabled = false; signingConfig = signingConfigs.getByName("debug") }
    }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    // Models and raw test tensors are memory-mapped straight from the APK.
    androidResources { noCompress += listOf("tflite", "f32") }
    packaging {
        // The Hexagon DSP loads its "skel" libraries from a real directory, so native libs must be
        // extracted on install (legacy packaging) rather than read from inside the APK.
        jniLibs {
            useLegacyPackaging = true
            // HTP (tensor accelerator) libs are ~180 MB and need Snapdragon 8 Gen 1-class chips;
            // this bench targets older Hexagon DSPs (v66) via the QNN DSP backend.
            excludes += listOf("**/libQnnHtp*.so")
        }
    }
}

dependencies {
    implementation("com.google.ai.edge.litert:litert:1.4.2")      // Interpreter, XNNPACK CPU, NNAPI delegate
    implementation("com.google.ai.edge.litert:litert-gpu:1.4.2")  // OpenCL/OpenGL GPU delegate
    implementation("com.qualcomm.qti:qnn-litert-delegate:2.52.0") // Qualcomm AI Engine Direct delegate
    implementation("com.qualcomm.qti:qnn-runtime:2.52.0")         // QNN backends (DSP v66, GPU)
}
