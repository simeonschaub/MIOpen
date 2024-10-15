/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#ifndef MIOPEN_GUARD_MLOPEN_READONLYRAMDB_HPP
#define MIOPEN_GUARD_MLOPEN_READONLYRAMDB_HPP

#include <miopen/db_record.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/stop_token.hpp>

#include <boost/optional.hpp>

#include <unordered_map>
#include <string>
#include <sstream>
#include <map>

namespace miopen {

struct DbPreloadStates;

namespace debug {
MIOPEN_INTERNALS_EXPORT bool& rordb_embed_fs_override();
} // namespace debug

class MIOPEN_INTERNALS_EXPORT ReadonlyRamDb
{
public:
    ReadonlyRamDb(DbKinds db_kind_, const fs::path& path) : db_kind(db_kind_), db_path(path) {}

    struct Instances
    {
        std::map<fs::path, std::unique_ptr<ReadonlyRamDb>> dbs;
        DbPreloadStates* states;
    };

    static auto GetInstances() -> Instances&;

    static ReadonlyRamDb& GetCached(DbKinds db_kind_,
                                    const fs::path& path,
                                    bool warn_if_unreadable,
                                    Instances& instances = GetInstances());

    boost::optional<DbRecord> FindRecord(const std::string& problem) const
    {
        MIOPEN_LOG_I2("Looking for key " << problem << " in file " << db_path);
        const auto it = cache.find(problem);

        if(it == cache.end())
            return boost::none;

        auto record = DbRecord{problem};

        MIOPEN_LOG_I2("Key match: " << problem);
        MIOPEN_LOG_I2("Contents found: " << it->second.content);

        if(!record.ParseContents(it->second.content))
        {
            MIOPEN_LOG_E("Error parsing payload under the key: "
                         << problem << " form file " << db_path << "#" << it->second.line);
            MIOPEN_LOG_E("Contents: " << it->second.content);
            return boost::none;
        }

        return record;
    }

    template <class TProblem>
    boost::optional<DbRecord> FindRecord(const TProblem& problem) const
    {
        const auto key = DbRecord::SerializeKey(db_kind, problem);
        return FindRecord(key);
    }

    template <class TProblem, class TValue>
    bool Load(const TProblem& problem, const std::string& id, TValue& value) const
    {
        const auto record = FindRecord(problem);
        if(!record)
            return false;
        return record->GetValues(id, value);
    }

    struct CacheItem
    {
        int line;
        std::string content;
    };

    const std::unordered_map<std::string, CacheItem>& GetCacheMap() const { return cache; }

    void Prefetch(bool warn_if_unreadable = true, stop_token stop = {});

private:
    DbKinds db_kind;
    fs::path db_path;
    std::unordered_map<std::string, CacheItem> cache;

    ReadonlyRamDb(const ReadonlyRamDb&) = default;
    ReadonlyRamDb(ReadonlyRamDb&&)      = default;
    ReadonlyRamDb& operator=(const ReadonlyRamDb&) = default;
    ReadonlyRamDb& operator=(ReadonlyRamDb&&) = default;

    void
    ParseAndLoadDb(std::istream& input_stream, bool warn_if_unreadable, stop_token const& stop);
};

} // namespace miopen

#endif
