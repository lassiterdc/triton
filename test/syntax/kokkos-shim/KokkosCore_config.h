/*
 * KEEP THIS FILE. The upstream tree deleted it when it integrated this branch's
 * work, and it is kept here deliberately. This note is also the edit that makes
 * a merge with upstream report a modify/delete conflict on this path rather than
 * drop the file with no conflict at all, because a path one side deleted and the
 * other side never changed merges as deleted. The conflict puts the deletion to
 * whoever resolves the merge instead of letting it be inherited.
 *
 * Why it is worth keeping: this is one of the six Kokkos configuration files that
 * test/syntax/run_syntax_check.py needs in order to parse TRITON's sources with
 * `mpic++ -fsyntax-only`, with no CMake configure, no Kokkos build and no link.
 * In a real build Kokkos' own CMake generates this file at configure time;
 * without the checked-in copy the parse stops inside Kokkos' headers before it
 * reaches any TRITON source, and that check is lost. That script's header says
 * why the check is worth keeping.
 */

#if !defined(KOKKOS_MACROS_HPP) || defined(KOKKOS_CORE_CONFIG_H)
#error "Do not include KokkosCore_config.h directly; include Kokkos_Macros.hpp instead."
#else
#define KOKKOS_CORE_CONFIG_H
#endif
#define KOKKOS_VERSION 40599
#define KOKKOS_VERSION_MAJOR 4
#define KOKKOS_VERSION_MINOR 5
#define KOKKOS_VERSION_PATCH 99
#define KOKKOS_ENABLE_SERIAL
#define KOKKOS_ENABLE_CXX17
#define KOKKOS_ENABLE_LIBDL
#define KOKKOS_ENABLE_COMPLEX_ALIGN
#define KOKKOS_ENABLE_DEPRECATED_CODE_4
#define KOKKOS_ENABLE_DEPRECATION_WARNINGS
#define KOKKOS_ARCH_NATIVE
