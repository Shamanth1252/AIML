import Image from 'next/image';
import Link from 'next/link';
import { getProductsByCategory, formatPrice, type Product } from '@/lib/products';

export default function CategoryPage({ params }: { params: { category: Product['category'] } }) {
	const products = getProductsByCategory(params.category);
	return (
		<div className="container py-10">
			<h1 className="text-3xl font-bold capitalize mb-6">{params.category}</h1>
			<div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
				{products.map((p) => (
					<Link key={p.id} href={`/${params.category}/${p.slug}`} className="group">
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