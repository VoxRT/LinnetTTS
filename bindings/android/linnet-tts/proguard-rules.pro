# LinnetTTS: the JNI layer resolves these methods by name. The `external fun`
# declarations live on the Kotlin object LinnetNative and are instance
# methods of that object, so the whole class (and its INSTANCE field) must
# survive shrinking and renaming.
-keepclasseswithmembers class com.voxrt.linnet.** { native <methods>; }
-keep class com.voxrt.linnet.LinnetNative { *; }
# Constructed from native code by class name and signature (int, String).
-keep class com.voxrt.linnet.LinnetException { <init>(int, java.lang.String); *; }
