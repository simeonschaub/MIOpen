/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017-2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef GUARD_TYPE_NAME_HPP
#define GUARD_TYPE_NAME_HPP

#include <string>
#include <string_view>
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__GNUC__)
#include <type_traits>
#endif

#include <miopen/config.h> // For BUILD_SHARED_LIBS and MIOPEN_ENABLE_FIN_INTERFACE

namespace miopen {

template <class T>
constexpr std::string_view constexpr_type_name()
{
#if defined(__clang__) || defined(__GNUC__)
    // clang or gcc
    constexpr auto full_name = std::string_view{__PRETTY_FUNCTION__};
#elif defined(_MSC_VER)
    // msvc
    constexpr auto full_name = std::string_view{__FUNCSIG__};
#endif

    // The substring with the data type name is located within the original string, between the prefix and the suffix, with the prefix always not at the beginning of the string and the suffix always at the end of the string.
#if defined(__clang__)
    // clang
    constexpr auto prefix = std::string_view{"[T = "};
    constexpr auto suffix = std::string_view{"]"};
#elif defined(__GNUC__)
    // gcc
    constexpr auto prefix = std::string_view{"[with T = "};
    constexpr auto suffix = std::string_view{"; std::string_view = std::basic_string_view<char>]"};
#elif defined(_MSC_VER)
    // msvc
    constexpr auto prefix = std::string_view{"constexpr_type_name<"};
    constexpr auto suffix = std::string_view{">(void)"};
#endif

    constexpr auto prefix_pos = full_name.find(prefix);
    static_assert(prefix_pos != std::string_view::npos);

    constexpr auto suffix_pos = full_name.rfind(suffix);
    static_assert(suffix_pos != std::string_view::npos);
    static_assert(suffix_pos == full_name.size() - suffix.size());

    constexpr auto pos = prefix_pos + prefix.size();
    static_assert(pos < suffix_pos);
    constexpr auto count = suffix_pos - pos;

    constexpr auto name = full_name.substr(pos, count);

#if defined(__clang__) || defined(__GNUC__)
    // clang or gcc
    return name;
#elif defined(_MSC_VER)
    // msvc
    if constexpr(std::is_compound_v<T>)
    {
        // For compound data types, the string contains the keyword 'class/struct/union/enum' before the data type name, separated by a space.
        constexpr auto sep = std::string_view{" "};
        constexpr auto sep_pos = name.find(sep);
        static_assert(sep_pos != std::string_view::npos);
        static_assert(sep_pos != 0); // must not be at the 0 position

        constexpr auto name_pos = sep_pos + sep.size();
        constexpr auto type_name = name.substr(name_pos);
        static_assert(type_name.size() > 0);

        return type_name;
    }
    else
    {
        return name;
    }
#endif
}

template <class T>
const std::string& get_type_name()
{
#if BUILD_SHARED_LIBS && MIOPEN_ENABLE_FIN_INTERFACE
    /// When using this function outside of the shared library, the static local variable is
    /// duplicated, both the library and the program using it have their own copy, but only one of
    /// them is initialized, depending on which entity calls the function first—the library or the
    /// program.
    /// \todo This needs to be removed when the interface matures, and internal class/function
    /// templates are no longer used by the fin.
    static std::string ret;
    if(ret.empty())
    {
        // The "new" operator is used here to avoid segmentation fault (since the variable is not
        // initialized).
        new(&ret) std::string(constexpr_type_name<T>());
    }
#else  // !BUILD_SHARED_LIBS || !MIOPEN_ENABLE_FIN_INTERFACE
    static const auto ret = std::string(constexpr_type_name<T>());
#endif // !BUILD_SHARED_LIBS || !MIOPEN_ENABLE_FIN_INTERFACE
    return ret;
}

template <class T>
const std::string& get_type_name(const T&)
{
    return miopen::get_type_name<T>();
}

} // namespace miopen

#endif
