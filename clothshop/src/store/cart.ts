import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type CartItem = {
	productId: string;
	title: string;
	priceCents: number;
	image: string;
	qty: number;
};

type CartState = {
	items: CartItem[];
	add: (item: Omit<CartItem, 'qty'>, qty?: number) => void;
	remove: (productId: string) => void;
	setQty: (productId: string, qty: number) => void;
	clear: () => void;
};

export const useCartStore = create<CartState>()(
	persist(
		(set, get) => ({
			items: [],
			add: (item, qty = 1) => {
				const existing = get().items.find((i) => i.productId === item.productId);
				if (existing) {
					set({
						items: get().items.map((i) =>
							i.productId === item.productId ? { ...i, qty: i.qty + qty } : i
						),
					});
				} else {
					set({ items: [...get().items, { ...item, qty }] });
				}
			},
			remove: (productId) => set({ items: get().items.filter((i) => i.productId !== productId) }),
			setQty: (productId, qty) => set({ items: get().items.map((i) => (i.productId === productId ? { ...i, qty } : i)) }),
			clear: () => set({ items: [] }),
		}),
		{ name: 'clothshop_cart' }
	)
);