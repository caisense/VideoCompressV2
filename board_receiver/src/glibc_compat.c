/* The SDK uses glibc 2.38 headers while the target image provides glibc 2.37.
 * C23 redirects strtol to __isoc23_strtol; bind that private executable symbol
 * to the legacy strtol ABI whose behavior is sufficient for CLI parsing. */
#if defined(__GNUC__)
#define BOARD_RECEIVER_HIDDEN __attribute__((visibility("hidden")))
#else
#define BOARD_RECEIVER_HIDDEN
#endif

extern long board_receiver_legacy_strtol(const char *text, char **end, int base)
    __asm__("strtol");

BOARD_RECEIVER_HIDDEN long __isoc23_strtol(const char *text, char **end, int base)
{
    return board_receiver_legacy_strtol(text, end, base);
}
