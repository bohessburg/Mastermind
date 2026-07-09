#include "v2/core/defs.h"
#include "v2/observe/card_text.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>

namespace {

void write_json_string(std::ostream& out, std::string_view text) {
    out << '"';
    for (const char ch : text) {
        switch (ch) {
        case '\\':
            out << "\\\\";
            break;
        case '"':
            out << "\\\"";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            out << ch;
            break;
        }
    }
    out << '"';
}

void write_type_list(std::ostream& out, std::uint16_t types) {
    struct TypeName {
        std::uint16_t bit;
        const char* name;
    };
    constexpr TypeName kTypes[] = {
        {TYPE_ACTION, "Action"},
        {TYPE_TREASURE, "Treasure"},
        {TYPE_VICTORY, "Victory"},
        {TYPE_CURSE, "Curse"},
        {TYPE_ATTACK, "Attack"},
        {TYPE_REACTION, "Reaction"},
        {TYPE_DURATION, "Duration"},
        {TYPE_NIGHT, "Night"},
        {TYPE_RESERVE, "Reserve"},
        {TYPE_COMMAND, "Command"},
    };

    out << '[';
    bool first = true;
    for (const TypeName& type : kTypes) {
        if ((types & type.bit) == 0U) {
            continue;
        }
        if (!first) {
            out << ',';
        }
        write_json_string(out, type.name);
        first = false;
    }
    out << ']';
}

bool dump_defs(const std::filesystem::path& output_path) {
    std::filesystem::create_directories(output_path.parent_path());

    std::ofstream out(output_path);
    if (!out) {
        return false;
    }

    out << "{\n  \"version\": 1,\n  \"defs\": [\n";
    const std::uint16_t count = card_def_count();
    for (DefId def = 0; def < count; ++def) {
        const CardDef& card = card_def(def);
        out << "    {\"id\": " << def << ", \"name\": ";
        write_json_string(out, card.name);
        out << ", \"cost\": {\"coins\": " << static_cast<int>(card.cost.coins)
            << ", \"potion\": " << static_cast<int>(card.cost.potion)
            << ", \"debt\": " << card.cost.debt
            << "}, \"types\": ";
        write_type_list(out, card.types);
        out << ", \"vp\": " << static_cast<int>(card.vp)
            << ", \"coin_value\": " << static_cast<int>(card.coin_value)
            << ", \"text\": ";
        write_json_string(out, card_text(def));
        out << '}';
        if (def + 1U < count) {
            out << ',';
        }
        out << '\n';
    }
    out << "  ]\n}\n";
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: v2_dump_defs <output-json>\n";
        return 2;
    }

    if (!dump_defs(argv[1])) {
        std::cerr << "failed to write " << argv[1] << '\n';
        return 1;
    }
    return 0;
}
