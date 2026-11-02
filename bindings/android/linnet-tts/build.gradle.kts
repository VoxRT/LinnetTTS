plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    `maven-publish`
}

val libGroup = project.property("GROUP") as String
val libVersion = project.property("VERSION_NAME") as String

android {
    namespace = "com.voxrt.linnet"
    compileSdk = libs.versions.compileSdk.get().toInt()

    defaultConfig {
        minSdk = libs.versions.minSdk.get().toInt()
        consumerProguardFiles("consumer-rules.pro")
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk {
            // Native libraries are produced by ../build-native.sh into src/main/jniLibs.
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    publishing {
        singleVariant("release") {
            withSourcesJar()
        }
    }
}

// Fail early with a readable message when the native libraries have not been built.
val checkNativeLibs by tasks.registering {
    val jniLibs = layout.projectDirectory.dir("src/main/jniLibs")
    doLast {
        val missing = listOf("arm64-v8a", "x86_64").filter { !jniLibs.file("$it/liblinnet_tts_jni.so").asFile.exists() }
        if (missing.isNotEmpty()) {
            throw GradleException(
                "liblinnet_tts_jni.so missing for ${missing.joinToString()}: run bindings/android/build-native.sh " +
                    "(needs an Android NDK, cargo-ndk and the Rust Android targets) before the Gradle build."
            )
        }
    }
}
tasks.matching { it.name.startsWith("merge") && it.name.endsWith("JniLibFolders") }.configureEach {
    dependsOn(checkNativeLibs)
}

dependencies {
    implementation(libs.androidx.core.ktx)
    androidTestImplementation(libs.androidx.test.ext.junit)
    androidTestImplementation(libs.androidx.test.runner)
}

publishing {
    publications {
        register<MavenPublication>("release") {
            groupId = libGroup
            artifactId = "linnet-tts"
            version = libVersion
            afterEvaluate { from(components["release"]) }
            pom {
                name.set("LinnetTTS for Android")
                description.set("On-device streaming text-to-speech: Kotlin API over the LinnetTTS Rust runtime")
                url.set("https://github.com/VoxRT/LinnetTTS")
                licenses {
                    license {
                        name.set("Apache-2.0")
                        url.set("https://www.apache.org/licenses/LICENSE-2.0")
                    }
                }
                scm {
                    url.set("https://github.com/VoxRT/LinnetTTS")
                }
            }
        }
    }
}
