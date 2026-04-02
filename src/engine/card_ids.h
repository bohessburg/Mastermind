#pragma once

#include "../game/card.h"

// Cached card definition IDs for fast integer comparison in agent hot paths.
// Call CardIds::init() after all cards are registered.
namespace CardIds {
    // Base treasures / victory / curse
    inline int Copper = -1;
    inline int Silver = -1;
    inline int Gold = -1;
    inline int Estate = -1;
    inline int Duchy = -1;
    inline int Province = -1;
    inline int Curse = -1;

    // Level 1 (Base Set)
    inline int Cellar = -1;
    inline int Chapel = -1;
    inline int Moat = -1;
    inline int Harbinger = -1;
    inline int Merchant = -1;
    inline int Vassal = -1;
    inline int Village = -1;
    inline int Workshop = -1;
    inline int Bureaucrat = -1;
    inline int Gardens = -1;
    inline int Laboratory = -1;
    inline int Market = -1;
    inline int Militia = -1;
    inline int Moneylender = -1;
    inline int Poacher = -1;
    inline int Remodel = -1;
    inline int Smithy = -1;
    inline int ThroneRoom = -1;
    inline int Bandit = -1;
    inline int CouncilRoom = -1;
    inline int Festival = -1;
    inline int Library = -1;
    inline int Mine = -1;
    inline int Sentry = -1;
    inline int Witch = -1;
    inline int Artisan = -1;

    // Level 2+
    inline int WorkersVillage = -1;
    inline int Courtyard = -1;
    inline int Hamlet = -1;
    inline int Lookout = -1;
    inline int Swindler = -1;
    inline int Scholar = -1;
    inline int KingsCourt = -1;
    inline int Oasis = -1;
    inline int Menagerie = -1;
    inline int Conspirator = -1;
    inline int Storeroom = -1;

    inline void init() {
        Copper   = CardRegistry::def_id_of("Copper");
        Silver   = CardRegistry::def_id_of("Silver");
        Gold     = CardRegistry::def_id_of("Gold");
        Estate   = CardRegistry::def_id_of("Estate");
        Duchy    = CardRegistry::def_id_of("Duchy");
        Province = CardRegistry::def_id_of("Province");
        Curse    = CardRegistry::def_id_of("Curse");

        Cellar       = CardRegistry::def_id_of("Cellar");
        Chapel       = CardRegistry::def_id_of("Chapel");
        Moat         = CardRegistry::def_id_of("Moat");
        Harbinger    = CardRegistry::def_id_of("Harbinger");
        Merchant     = CardRegistry::def_id_of("Merchant");
        Vassal       = CardRegistry::def_id_of("Vassal");
        Village      = CardRegistry::def_id_of("Village");
        Workshop     = CardRegistry::def_id_of("Workshop");
        Bureaucrat   = CardRegistry::def_id_of("Bureaucrat");
        Gardens      = CardRegistry::def_id_of("Gardens");
        Laboratory   = CardRegistry::def_id_of("Laboratory");
        Market       = CardRegistry::def_id_of("Market");
        Militia      = CardRegistry::def_id_of("Militia");
        Moneylender  = CardRegistry::def_id_of("Moneylender");
        Poacher      = CardRegistry::def_id_of("Poacher");
        Remodel      = CardRegistry::def_id_of("Remodel");
        Smithy       = CardRegistry::def_id_of("Smithy");
        ThroneRoom   = CardRegistry::def_id_of("Throne Room");
        Bandit       = CardRegistry::def_id_of("Bandit");
        CouncilRoom  = CardRegistry::def_id_of("Council Room");
        Festival     = CardRegistry::def_id_of("Festival");
        Library      = CardRegistry::def_id_of("Library");
        Mine         = CardRegistry::def_id_of("Mine");
        Sentry       = CardRegistry::def_id_of("Sentry");
        Witch        = CardRegistry::def_id_of("Witch");
        Artisan      = CardRegistry::def_id_of("Artisan");

        WorkersVillage = CardRegistry::def_id_of("Worker's Village");
        Courtyard      = CardRegistry::def_id_of("Courtyard");
        Hamlet         = CardRegistry::def_id_of("Hamlet");
        Lookout        = CardRegistry::def_id_of("Lookout");
        Swindler       = CardRegistry::def_id_of("Swindler");
        Scholar        = CardRegistry::def_id_of("Scholar");
        KingsCourt     = CardRegistry::def_id_of("King's Court");
        Oasis          = CardRegistry::def_id_of("Oasis");
        Menagerie      = CardRegistry::def_id_of("Menagerie");
        Conspirator    = CardRegistry::def_id_of("Conspirator");
        Storeroom      = CardRegistry::def_id_of("Storeroom");
    }
}
