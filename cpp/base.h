/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file base.h
 */

#ifndef MLC_LLM_DLL
#ifdef _WIN32
#ifdef MLC_LLM_EXPORTS
#define MLC_LLM_DLL __declspec(dllexport)
#else
#define MLC_LLM_DLL __declspec(dllimport)
#endif
#else
#define MLC_LLM_DLL __attribute__((visibility("default")))
#endif
#endif
