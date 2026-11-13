#include "v2/observe/card_text.h"

#include "v2/core/defs.h"

namespace {

constexpr const char* kCardText[] = {
    "+1 Coin.",
    "+2 Coins.",
    "+3 Coins.",
    "+5 Coins.",
    "+1 Potion.",
    "1 Victory Point.",
    "3 Victory Points.",
    "6 Victory Points.",
    "10 Victory Points.",
    "-1 Victory Point.",
    "+1 Action. Discard any number of cards, then draw that many.",
    "Trash up to 4 cards from your hand.",
    "+1 Card. +2 Actions.",
    "+3 Cards.",
    "Gain a card costing up to 4 Coins.",
    "Trash a card from your hand. Gain a card costing up to 2 Coins more than it.",
    "You may trash a Treasure from your hand. Gain a Treasure to your hand costing up to 3 Coins more than it.",
    "Test-only card: choose exactly two cards to trash.",
    "Test-only card: repeat a program containing a choice.",
    "+1 Card. +1 Action. The first time you play a Silver this turn, +1 Coin.",
    "+2 Coins. Each other player discards down to 3 cards in hand.",
    "+2 Cards. Each other player gains a Curse.",
    "+2 Cards. When another player plays an Attack card, you may first reveal this from your hand to be unaffected by it.",
    "Gain a Silver onto your deck. Each other player reveals a Victory card from their hand and puts it onto their deck, or reveals a hand with no Victory cards.",
    "Test-only trigger card: alpha ordering probe.",
    "Test-only trigger card: beta ordering probe.",
    "Test-only trigger card: gamma ordering probe.",
    "+1 Card. +1 Action. +1 Buy. +1 Coin.",
    "+2 Actions. +1 Buy. +2 Coins.",
    "+2 Cards. +1 Action.",
    "Worth 1 Victory Point per 10 cards you have, rounded down.",
    "You may trash a Copper from your hand for +3 Coins.",
    "+1 Card. +1 Action. +1 Coin. Discard a card per empty Supply pile.",
    "+2 Coins. Discard the top card of your deck. If it is an Action card, you may play it.",
    "+1 Card. +1 Action. Look through your discard pile. You may put a card from it onto your deck.",
    "You may play an Action card from your hand twice.",
    "+4 Cards. +1 Buy. Each other player draws a card.",
    "Gain a card to your hand costing up to 5 Coins. Put a card from your hand onto your deck.",
    "Gain a Gold. Each other player reveals the top 2 cards of their deck, trashes a revealed Treasure other than Copper, and discards the rest.",
    "Draw until you have 7 cards in hand, skipping any Action cards you choose to; set those aside, discarding them afterwards.",
    "+1 Card. +1 Action. Look at the top 2 cards of your deck. Trash and/or discard any number of them. Put the rest back on top in any order.",
};

static_assert(sizeof(kCardText) / sizeof(kCardText[0]) == BASIC_CARD_COUNT);

} // namespace

const char* card_text(DefId def) noexcept {
    if (def >= BASIC_CARD_COUNT) {
        return "";
    }
    return kCardText[def];
}
