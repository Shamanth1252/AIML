"use client";
import { Button } from '@/components/ui/Button';
import { useCartStore } from '@/store/cart';

export function AddToCartButton(props: { productId: string; title: string; priceCents: number; image: string }) {
	const add = useCartStore((s) => s.add);
	return (
		<Button onClick={() => add(props)} className="w-full md:w-auto">Add to cart</Button>
	);
}