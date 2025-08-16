import Image from 'next/image';
import { notFound } from 'next/navigation';
import { findProductBySlug, formatPrice, type Product } from '@/lib/products';
import { AddToCartButton } from '@/components/cart/AddToCartButton';

export default function ProductPage({ params }: { params: { category: Product['category']; slug: string } }) {
	const product = findProductBySlug(params.slug);
	if (!product) return notFound();

	return (
		<div className="container py-10 grid md:grid-cols-2 gap-10">
			<div className="relative aspect-square overflow-hidden rounded-2xl bg-neutral-100">
				<Image src={product.image} alt={product.title} fill className="object-cover"/>
			</div>
			<div className="space-y-6">
				<div>
					<div className="text-sm text-neutral-500 capitalize">{product.category}</div>
					<h1 className="text-3xl font-semibold mt-1">{product.title}</h1>
				</div>
				<p className="text-neutral-600 max-w-prose">{product.description}</p>
				<div className="text-xl font-semibold">{formatPrice(product.priceCents)}</div>
				<AddToCartButton productId={product.id} title={product.title} priceCents={product.priceCents} image={product.image} />
			</div>
		</div>
	);
}