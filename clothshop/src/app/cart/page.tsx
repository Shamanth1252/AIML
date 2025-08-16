"use client";
import Image from 'next/image';
import Link from 'next/link';
import { useCartStore } from '@/store/cart';
import { formatPrice } from '@/lib/products';
import { Button } from '@/components/ui/Button';

export default function CartPage() {
	const { items, remove, setQty, clear } = useCartStore();
	const subtotal = items.reduce((acc, i) => acc + i.priceCents * i.qty, 0);

	return (
		<div className="container py-10 grid md:grid-cols-[1fr_360px] gap-10">
			<div>
				<h1 className="text-3xl font-bold mb-6">Your bag</h1>
				{items.length === 0 ? (
					<div className="text-neutral-600">Your bag is empty. <Link href="/" className="underline">Continue shopping</Link></div>
				) : (
					<ul className="space-y-6">
						{items.map((i) => (
							<li key={i.productId} className="flex gap-4">
								<div className="relative w-28 h-28 rounded-lg overflow-hidden bg-neutral-100">
									<Image src={i.image} alt={i.title} fill className="object-cover"/>
								</div>
								<div className="flex-1">
									<div className="font-medium">{i.title}</div>
									<div className="text-sm text-neutral-600">{formatPrice(i.priceCents)}</div>
									<div className="mt-2 flex items-center gap-2">
										<label htmlFor={`qty-${i.productId}`} className="text-sm">Qty</label>
										<select id={`qty-${i.productId}`} value={i.qty} onChange={(e) => setQty(i.productId, Number(e.target.value))} className="border rounded px-2 py-1">
											{[1,2,3,4,5].map(n => <option key={n} value={n}>{n}</option>)}
										</select>
										<button onClick={() => remove(i.productId)} className="text-sm underline">Remove</button>
									</div>
								</div>
							</li>
						))}
					</ul>
				)}
			</div>
			<aside className="rounded-2xl border p-6 h-fit">
				<div className="flex justify-between">
					<span>Subtotal</span>
					<span className="font-medium">{formatPrice(subtotal)}</span>
				</div>
				<p className="text-xs text-neutral-500 mt-2">Taxes and shipping calculated at checkout.</p>
				<div className="mt-4 space-y-3">
					<Link href="/checkout" className="block"><Button className="w-full">Checkout</Button></Link>
					<Button variant="secondary" className="w-full" onClick={() => clear()}>Clear bag</Button>
				</div>
			</aside>
		</div>
	);
}