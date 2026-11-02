package com.voxrt.linnet

/**
 * Error reported by the native runtime. [code] follows the C API (`LINNET_ERR_*`):
 * -1 invalid argument, -2 I/O, -3 file format, -4 model shape, -6 internal.
 */
class LinnetException(val code: Int, message: String) : RuntimeException(message) {
    companion object {
        const val INVALID_ARG = -1
        const val IO = -2
        const val FORMAT = -3
        const val SHAPE = -4
        const val INTERNAL = -6
    }
}
