#pragma once

// Well-known card def_ids for O(1) identity checks.
// Call CardDefs::init() once after all cards are registered.
namespace CardDefs {
    extern int COPPER;
    extern int SILVER;
    extern int GOLD;
    extern int ESTATE;
    extern int DUCHY;
    extern int PROVINCE;
    extern int CURSE;

    void init();
}
