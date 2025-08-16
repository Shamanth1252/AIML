"use client";
import Link from 'next/link';
import { useCartStore } from '@/store/cart';
import { formatPrice } from '@/lib/products';
import { Button } from '@/components/ui/Button';
import { useState } from 'react';

export default function CheckoutPage() {
	const { items, clear } = useCartStore();
	const subtotal = items.reduce((acc, i) => acc + i.priceCents * i.qty, 0);
	const [placing, setPlacing] = useState(false);

	async function placeOrder() {
		setPlacing(true);
		await new Promise((r) => setTimeout(r, 800));
		clear();
		setPlacing(false);
		alert('Order placed! This is a demo.');
	}

	return (
		<div className="container py-10">
			<h1 className="text-3xl font-bold mb-6">Checkout</h1>
			{items.length === 0 ? (
				<div>Your bag is empty. <Link href="/" className="underline">Shop now</Link></div>
			) : (
				<div className="max-w-xl space-y-4">
					<ul className="divide-y border rounded-lg">
						{items.map((i) => (
							<li key={i.productId} className="p-4 flex justify-between">
								<span>{i.title} × {i.qty}</span>
								<span>{formatPrice(i.priceCents * i.qty)}</span>
							</li>
						))}
						<li className="p-4 flex justify-between font-medium">
							<span>Total</span>
							<span>{formatPrice(subtotal)}</span>
						</li>
					</ul>
					<Button onClick={placeOrder} disabled={placing} className="w-full">{placing ? 'Placing...' : 'Place order'}</Button>
					<p className="text-xs text-neutral-500">This demo does not process payments.</p>
				</div>
			)}
		</div>
	);
}