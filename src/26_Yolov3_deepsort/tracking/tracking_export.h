
#ifndef TRACKING_EXPORT_H
#define TRACKING_EXPORT_H

#ifdef TRACKING_STATIC_DEFINE
#  define TRACKING_EXPORT
#  define TRACKING_NO_EXPORT
#else
#  ifndef TRACKING_EXPORT
#    ifdef tracking_EXPORTS
        /* We are building this library */
#      define TRACKING_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define TRACKING_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef TRACKING_NO_EXPORT
#    define TRACKING_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef TRACKING_DEPRECATED
#  define TRACKING_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef TRACKING_DEPRECATED_EXPORT
#  define TRACKING_DEPRECATED_EXPORT TRACKING_EXPORT TRACKING_DEPRECATED
#endif

#ifndef TRACKING_DEPRECATED_NO_EXPORT
#  define TRACKING_DEPRECATED_NO_EXPORT TRACKING_NO_EXPORT TRACKING_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef TRACKING_NO_DEPRECATED
#    define TRACKING_NO_DEPRECATED
#  endif
#endif

#endif /* TRACKING_EXPORT_H */
