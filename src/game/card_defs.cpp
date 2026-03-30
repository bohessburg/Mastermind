#include "card_defs.h"
#include "card.h"

namespace CardDefs {

int COPPER   = -1;
int SILVER   = -1;
int GOLD     = -1;
int ESTATE   = -1;
int DUCHY    = -1;
int PROVINCE = -1;
int CURSE    = -1;

void init() {
    COPPER   = CardRegistry::get_id("Copper");
    SILVER   = CardRegistry::get_id("Silver");
    GOLD     = CardRegistry::get_id("Gold");
    ESTATE   = CardRegistry::get_id("Estate");
    DUCHY    = CardRegistry::get_id("Duchy");
    PROVINCE = CardRegistry::get_id("Province");
    CURSE    = CardRegistry::get_id("Curse");
}

} // namespace CardDefs
