import generatedDefs from '../../client-data/defs.gen.json';
import type { CardDef, DefTable } from './protocol';

export const defTable = generatedDefs as DefTable;

export const defsById = new Map<number, CardDef>(
  defTable.defs.map((def) => [def.id, def]),
);

export const defsByName = new Map<string, CardDef>(
  defTable.defs.map((def) => [def.name, def]),
);

export function cardDef(defId: number | null | undefined): CardDef | undefined {
  if (defId === null || defId === undefined) {
    return undefined;
  }
  return defsById.get(defId);
}

export function kingdomPreset(): string[] {
  return [
    'Sentry',
    'Library',
    'Throne Room',
    'Bandit',
    'Witch',
    'Moat',
    'Village',
    'Smithy',
    'Market',
    'Remodel',
  ];
}
