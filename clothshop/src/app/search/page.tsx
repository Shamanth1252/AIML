"use client";
import { useMemo, useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { PRODUCTS, formatPrice, type Product } from '@/lib/products';

export default function SearchPage() {
	const [q, setQ] = useState('');
	const [cat, setCat] = useState<Product['category'] | 'all'>('all');

	const results = useMemo(() => {
		return PRODUCTS.filter((p) => {
			const matchesQ = q ? (p.title + ' ' + p.description).toLowerCase().includes(q.toLowerCase()) : true;
			const matchesCat = cat === 'all' ? true : p.category === cat;
			return matchesQ && matchesCat;
		});
	}, [q, cat]);

	return (
		<div className="container py-10">
			<h1 className="text-3xl font-bold mb-6">Search</h1>
			<div className="flex gap-3 mb-6">
				<input className="border rounded px-3 py-2 flex-1" placeholder="Search products" value={q} onChange={(e) => setQ(e.target.value)} />
				<select className="border rounded px-3 py-2" value={cat} onChange={(e) => setCat(e.target.value as any)}>
					<option value="all">All</option>
					<option value="men">Men</option>
					<option value="women">Women</option>
					<option value="kids">Kids</option>
				</select>
			</div>
			<div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
				{results.map((p) => (
					<Link key={p.id} href={`/${p.category}/${p.slug}`} className="group">
						<div className="relative aspect-square overflow-hidden rounded-xl bg-neutral-100">
							<Image src={p.image} alt={p.title} fill className="object-cover transition-transform duration-500 group-hover:scale-105"/>
						</div>
						<div className="mt-3 space-y-1">
							<div className="text-sm text-neutral-500">{p.category}</div>
							<div className="font-medium">{p.title}</div>
							<div className="text-sm">{formatPrice(p.priceCents)}</div>
						</div>
					</Link>
				))}
			</div>
		</div>
	);
}